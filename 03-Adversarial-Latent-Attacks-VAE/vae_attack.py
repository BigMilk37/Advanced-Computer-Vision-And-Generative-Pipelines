import torch
import warnings
from diffusers import AutoencoderKL
from torch.nn.functional import pad 
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider' is not in available provider names")

from attacks.base import Attack

class VAEAttack(Attack):

    
    def __init__(
        self,
        n_avg_imgs: int = 1,            
        noise_level: float = 0.05,      
        device: str = "auto",
        cache_dir: str | None = None,
        vae_name: str = "stabilityai/sd-vae-ft-mse" 
    ) -> None:
        super().__init__()

        self.n_avg_imgs = 1
        self.noise_level = 0.05
        

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu") 
        elif device == "mps":
             print("Warning: 'mps' device requested. Overriding to 'cpu' to avoid VAE padding bugs.")
             self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"VAE Attack Config: Device={self.device}, Noise={self.noise_level} (v5.0)")
        

        print(f"Loading VAE: {vae_name}...")
        try:
            self.vae = AutoencoderKL.from_pretrained(vae_name, cache_dir=cache_dir)
            print(">>> SUCCESS: Loaded stabilityai/sd-vae-ft-mse")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load {vae_name}")
            self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", cache_dir=cache_dir)

        self.vae.to(self.device).eval()
        self.latent_scale_factor = self.vae.config.scaling_factor if hasattr(self.vae.config, "scaling_factor") else 0.18215

    def add_noise_to_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.noise_level <= 0.0:
            return embeddings

        std = embeddings.std(dim=(2, 3), keepdim=True)
        std = torch.where(std > 0, std, torch.tensor(1.0, device=embeddings.device, dtype=embeddings.dtype))
        noise = torch.randn_like(embeddings) * std * self.noise_level
        return embeddings + noise

    def match_stats(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Restores the mean and standard deviation of the target image."""
        mu_s = source.mean(dim=(2, 3), keepdim=True)
        std_s = source.std(dim=(2, 3), keepdim=True)
        mu_t = target.mean(dim=(2, 3), keepdim=True)
        std_t = target.std(dim=(2, 3), keepdim=True)
        
        return (source - mu_s) * (std_t / (std_s + 1e-6)) + mu_t

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        img = img.to(self.device)
        
        min_val, max_val = img.min(), img.max()
        if max_val > 1.0: img_0_1 = img / 255.0
        elif min_val < 0.0: img_0_1 = (img + 1.0) / 2.0
        else: img_0_1 = img

        original_0_1 = img_0_1.clone()
        normalized_img = 2.0 * img_0_1 - 1.0

        H, W = normalized_img.shape[-2], normalized_img.shape[-1]
        pad_H = (8 - (H % 8)) % 8
        pad_W = (8 - (W % 8)) % 8
        
        if pad_H > 0 or pad_W > 0:
            normalized_img = pad(normalized_img, (0, pad_W, 0, pad_H), mode='constant', value=0.0)

        accumulated_output = torch.zeros_like(normalized_img)


        with torch.no_grad():
            dist = self.vae.encode(normalized_img).latent_dist
            base_latents = dist.mode() * self.latent_scale_factor
            

            noisy_latents = self.add_noise_to_embeddings(base_latents)
            vae_purified = self.vae.decode(noisy_latents / self.latent_scale_factor).sample
            vae_purified = torch.clamp(vae_purified, -1, 1)


        if pad_H > 0 or pad_W > 0:
            vae_purified = vae_purified[:, :, :H, :W]


        final_output = (vae_purified * 0.5) + 0.5
        final_output = self.match_stats(final_output, original_0_1)
        final_output = torch.clamp(final_output, 0, 1)
        
        return final_output