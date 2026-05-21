import numpy as np
import torch
from tqdm import tqdm

from attacks.base import Attack

from .model_dip import get_net_dip


def get_model(dig_cfgs):

    if dig_cfgs["arch"] == "vanila":
        dip_model = get_net_dip(dig_cfgs["arch"])
    else:
        raise RuntimeError("Unsupported DIP architecture.")
    dip_model.train()
    return dip_model


class DIPAttack(Attack):
    """
    DIP-based watermark evasion attack.
    
    Deep Image Prior (DIP) is a technique that uses the structure of a convolutional
    neural network as a prior for natural images. This attack optimizes a randomly
    initialized network to reconstruct the input image, which can remove watermarks
    while preserving the main content.
    
    The key insight is that the network architecture itself acts as a regularization
    that favors natural images over watermarks, which are often more structured.
    
    NOTE: This version uses the original image as input (non-randomized), which is
    slightly different from the original DIP paper but is commonly used in practice.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float32,
        total_iters: int = 100,
        lr: float = 0.005,
        arch: str = "vanila",
    ) -> None:
        """
        Initialize DIP attack.

        Args:
            device: Device to run computations on (CPU or CUDA)
            dtype: Data type for computations (float32 or float16)
            total_iters: Total number of optimization iterations
            lr: Learning rate for the optimizer
            arch: DIP architecture type (currently only "vanila" is supported)
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.total_iters = total_iters
        self.lr = lr
        self.arch = arch

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Perform DIP-based attack on input image(s).
        
        Args:
            img: Input image tensor, shape (batch, channels, height, width) or (channels, height, width)
                 Values should be in range [0, 1]
                 
        Returns:
            Attacked image tensor with same shape as input, values in range [0, 1]
        """
        # Ensure input is 4D (batch, C, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)
            
        batch_size, c, h, w = img.shape
        results = []
        

        for i in range(batch_size):
            single_img = img[i:i+1].to(self.device, dtype=self.dtype)
            

            net = get_model({"arch": self.arch}).to(self.device, dtype=self.dtype)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            criterion = torch.nn.MSELoss()
            

            net_input = single_img.clone().detach()
            
            reg_noise_std = 0.01
            

            iterator = range(self.total_iters)
            
            for _ in iterator:
                optimizer.zero_grad()
                

                noise = torch.randn_like(net_input) * reg_noise_std
                
                output = net(net_input + noise)
                loss = criterion(output, single_img)
                loss.backward()
                optimizer.step()
                
            results.append(output.detach())
            
        return torch.cat(results, dim=0)


def fill_noise(x, noise_type):

    if noise_type == "u":
        x.uniform_()
    elif noise_type == "n":
        x.normal_()
    else:
        assert False


def np_to_torch(img_np):

    return torch.from_numpy(img_np)[None, :]


def get_noise(input_depth, method, spatial_size, noise_type="u", var=1.0 / 10):

    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == "noise":
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == "meshgrid":
        assert input_depth == 2
        X, Y = np.meshgrid(
            np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1), 
            np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1)
        )
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


class DIPAttackNoise(Attack):

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float32,
        total_iters: int = 200,
        lr: float = 0.003,
        arch: str = "vanila",
        input_noise_method: str = "n",
        input_noise_var: float = 1.0 / 10,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.total_iters = total_iters
        self.lr = lr
        self.arch = arch
        self.input_noise_method = input_noise_method
        self.input_noise_var = input_noise_var

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        if img.dim() == 3:
            img = img.unsqueeze(0)
            
        batch_size, c, h, w = img.shape
        results = []
        
        for i in range(batch_size):
            single_img = img[i:i+1].to(self.device, dtype=self.dtype)
            

            net = get_model({"arch": self.arch}).to(self.device, dtype=self.dtype)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            criterion = torch.nn.MSELoss()
            
            net_input = get_noise(
                c, 
                "noise", 
                (h, w), 
                self.input_noise_method, 
                self.input_noise_var
            ).to(self.device, dtype=self.dtype)
            
            reg_noise_std = 0.03

            iterator = range(self.total_iters)
            
            for _ in iterator:
                optimizer.zero_grad()
                

                net_input_perturbed = net_input + (torch.randn_like(net_input) * reg_noise_std)
                
                output = net(net_input_perturbed)
                
                loss = criterion(output, single_img)
                loss.backward()
                optimizer.step()
                
            results.append(output.detach())
            
        return torch.cat(results, dim=0)