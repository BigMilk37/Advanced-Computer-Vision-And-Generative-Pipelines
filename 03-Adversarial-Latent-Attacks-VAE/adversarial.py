import numpy as np
import torch
from tqdm import tqdm

from attacks.base import Attack

from .feature_extractors import ClipEmbedding, ResNet18Embedding


class AdversarialEmbedding(Attack):


    def __init__(self,
                 device: torch.device | str,
                 encoder: str,
                 loss_type: str = "l2",  
                 strength: int = 4, 
                 eps_factor: float = 1 / 255,
                 alpha_factor: float = 0.1,
                 n_steps: int = 200,
                 random_start: bool = True,
                 ) -> None:
        super().__init__()


        self.encoder = encoder

        if encoder == "resnet18":

            embedding_model = ResNet18Embedding("last", device)
        elif encoder == "clip":
            embedding_model = ClipEmbedding(device)
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

        embedding_model = embedding_model.to(device)
        embedding_model.eval()

        self.model = embedding_model
        self.device = device
        self.eps = eps_factor * strength  
        self.alpha = alpha_factor * self.eps  
        self.steps = n_steps
        self.loss_type = loss_type
        self.random_start = random_start

        if self.loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif self.loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    def pgd(self, images: torch.Tensor, init_delta: torch.Tensor = None) -> torch.Tensor:
        """
        Perform Projected Gradient Descent (PGD) attack.
        
        Args:
            images: Input images, shape (batch, channels, height, width), values in [0, 1]
            init_delta: Optional initial perturbation (if not using random start)
            
        Returns:
            Adversarial images with same shape as input, values in [0, 1]
        """
        images = images.clone().detach().to(self.device)
        
        with torch.no_grad():
            orig_embeddings = self.model(images)

        delta = torch.zeros_like(images).to(self.device)
        
        if init_delta is not None:
            delta = init_delta.to(self.device)
        elif self.random_start:
            delta.uniform_(-self.eps, self.eps)
            
        delta.requires_grad = True

        for _ in range(self.steps):
            adv_images = torch.clamp(images + delta, 0, 1)
            adv_embeddings = self.model(adv_images)
            loss = self.loss_fn(adv_embeddings, orig_embeddings)
            grad = torch.autograd.grad(loss, delta)[0]
            
            delta.data = delta.data + self.alpha * torch.sign(grad)
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images
        return torch.clamp(images + delta.detach(), 0, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) < 4:
            img = img.unsqueeze(0)

        return self.pgd(img)