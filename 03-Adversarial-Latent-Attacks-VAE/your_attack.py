import os, sys
import argparse
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToPILImage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR + "/watermarks/arwgan/")

try:
    from datasets import ImageDataset
    from watermarks import Stegastamp, StableSignatureDecoder, ARWGAN
    from metrics import BitAccuracy, PSNR, VMAF, LPIPS
except ImportError as e:
    print(f"Critical Import Error: {e}")
    print("Ensure you are running this in the correct environment with 'watermarks', 'datasets', and 'metrics' packages.")
    sys.exit(1)

to_pil = ToPILImage()

class Attack(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
class YourAttack(Attack):
    def __init__(self, 
                 device: torch.device | str,
                 tv_weight: float = 0.08,
                 n_steps: int = 150,
                 lr: float = 0.05):
        super().__init__()
        self.device = device
        self.tv_weight = tv_weight
        self.n_steps = n_steps
        self.lr = lr

    def total_variation_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h_x-1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w_x-1]), 2).sum()
        return h_tv + w_tv

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) < 4: img = img.unsqueeze(0)
        img = img.detach().to(self.device)
        adv = img.clone().requires_grad_(True)
        optimizer = optim.Adam([adv], lr=self.lr)
        
        for _ in range(self.n_steps):
            optimizer.zero_grad()
            l2_loss = nn.MSELoss()(adv, img)
            tv_loss = self.total_variation_loss(adv)
            loss = l2_loss + self.tv_weight * tv_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                adv.data.clamp_(0, 1)
        return adv.detach()

class ResizeAttack(Attack):
    def __init__(self, device, scale_min=0.7, scale_max=0.8):
        super().__init__()
        self.device = device
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        N, C, H, W = img.shape
        scale = np.random.uniform(self.scale_min, self.scale_max)
        new_H, new_W = int(H * scale), int(W * scale)
        
        downscaled = F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=False)
        restored = F.interpolate(downscaled, size=(H, W), mode='bilinear', align_corners=False)
        return restored

class SharpeningFilter(Attack):
    def __init__(self, device, factor=1.5):
        super().__init__()
        self.device = device
        self.factor = factor
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 1.0
        self.kernel = kernel.repeat(3, 1, 1, 1).to(device)

    def forward(self, img):
        sharpened = F.conv2d(img, self.kernel, padding=1, groups=3)
        return torch.clamp(sharpened, 0, 1)


class CombinedAttack(Attack):
    def __init__(self, device, scale=0.75):
        super().__init__()
        self.resize = ResizeAttack(device, scale_min=scale, scale_max=scale)
        self.sharpen = SharpeningFilter(device, factor=1.0)
        
    def forward(self, img):
        img = self.resize(img)
        
        steps = 32.0 # 2^5
        img = (img * steps).round() / steps
        
        img = self.sharpen(img)
        
        return img



WATERMARKS = {
    "ARWGAN": (ARWGAN, {"device": "cpu"}),
    "StegaStamp": (Stegastamp, {"device": "cpu"}),
    "StableSignature": (StableSignatureDecoder, {"device": "cpu"})
}

ATTACKS = {
    "combined": (CombinedAttack, {"scale": 0.75}), 
    "resize": (ResizeAttack, {"scale_min": 0.7, "scale_max": 0.8}),
    "your": (YourAttack, {}),
}

def device_or_default(device_arg):
    if device_arg: return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"

def run_one(watermark_name, attack_name, path, device, save_images_dir=None, out_json=None):
    if watermark_name not in WATERMARKS: raise ValueError(f"Unknown watermark: {watermark_name}")
    if attack_name not in ATTACKS: raise ValueError(f"Unknown attack: {attack_name}")

    device = device_or_default(device)

    # Initialize Metrics
    bit_accur = BitAccuracy()
    psnr = PSNR()
    vmaf = VMAF()
    lpips = LPIPS().to(device)

    # Initialize Watermark
    w_cls, w_kwargs = WATERMARKS[watermark_name]
    w_kwargs = dict(w_kwargs)
    w_kwargs["device"] = device
    watermark = w_cls(**w_kwargs)

    # Initialize Attack
    a_cls, a_kwargs = ATTACKS[attack_name]
    a_kwargs = dict(a_kwargs)
    a_kwargs["device"] = device
    attack = a_cls(**a_kwargs)

    imgs = ImageDataset(path)
    results = {}
    if out_json: os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    if save_images_dir: os.makedirs(save_images_dir, exist_ok=True)

    quality_score_s = []
    bit_accur_s = []

    print(f"Running {attack_name} attack on {watermark_name}...")

    for img in tqdm(imgs, desc=f"{watermark_name}:{attack_name}"):
        fname = img[0]
        orig_tensor = img[1].to(device)

        true_msg = watermark.decode(orig_tensor.unsqueeze(0).to("cpu"))
        attacked = attack.forward(orig_tensor).to(device, dtype=torch.float)
        attacked_msg = watermark.decode(attacked.to("cpu"))

        bit_accur_v = float(bit_accur(attacked_msg, true_msg).item())
        psnr_v = float(psnr(attacked, orig_tensor.unsqueeze(0)).item())
        vmaf_v = float(vmaf(attacked.to("cpu"), orig_tensor.unsqueeze(0).to("cpu")).item())
        lpips_v = float(lpips(attacked, orig_tensor.unsqueeze(0)).item())

        quality_score = 0.5 + (-2 * psnr_v + 400 * lpips_v - 1 * vmaf_v) / 1000
        quality_score_s.append(quality_score)
        bit_accur_s.append(bit_accur_v)

        img_res = {
            "true_msg": true_msg.tolist(),
            f"{attack_name}_msg": attacked_msg.tolist(),
            f"{attack_name}_bit_accur": bit_accur_v,
            f"{attack_name}_psnr": psnr_v,
            f"{attack_name}_vmaf": vmaf_v,
            f"{attack_name}_lpips": lpips_v
        }
        results[fname] = img_res

        if save_images_dir:
            try:
                pil = to_pil(attacked.clamp(0, 1).cpu()[0])
                out_path = os.path.join(save_images_dir, f"{watermark_name}__{attack_name}__{os.path.basename(fname)}")
                pil.save(out_path)
            except Exception as e:
                print(f"Warning: failed to save image {fname}: {e}")

    wm_performance = sum(bit_accur_s) / len(bit_accur_s) if bit_accur_s else 0
    quality_degradation = sum(quality_score_s) / len(quality_score_s) if quality_score_s else 0
    score = (wm_performance ** 2 + quality_degradation ** 2) ** 0.5

    results["Watermark Performance"] = wm_performance
    results["Quality Degradation"] = quality_degradation
    results["Score"] = score

    if out_json:
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved metrics to {out_json}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermark", required=True, help="Watermark name. Choices: " + ", ".join(WATERMARKS.keys()))
    parser.add_argument("--attack", required=True, help="Attack name. Choices: " + ", ".join(ATTACKS.keys()))
    parser.add_argument("--path", required=False, help="Folder with test images.", default=None)
    parser.add_argument("--device", required=False, help="device", default=None)
    parser.add_argument("--out", required=False, help="JSON file.", default=f"./results_placeholder.json")
    parser.add_argument("--save-images", required=False, help="Directory to save images.", default=None)
    args = parser.parse_args()

    if args.path is None: args.path = f"./data/test/{args.watermark}/"
    if args.out and "{watermark}" in args.out: out_json = args.out.format(watermark=args.watermark, attack=args.attack)
    else: out_json = f"./results/{args.watermark}__{args.attack}.json" if "placeholder" in args.out else args.out

    results = run_one(args.watermark, args.attack, args.path, args.device, save_images_dir=args.save_images, out_json=out_json)
    print("Done. Processed", len(results) - 3, "images.")
    print(f"Watermark Performance: {results['Watermark Performance']:.4f}")
    print(f"Quality Degradation: {results['Quality Degradation']:.4f}")
    print(f"Score: {results['Score']:.4f}")

if __name__ == "__main__":
    main()