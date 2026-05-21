import torch
import torchvision
from torchvision import transforms
from transformers import AutoProcessor, CLIPModel


class ClipEmbedding(torch.nn.Module):
    
    def __init__(self, device):
        """
        Initialize CLIP embedding model.
        
        Args:
            device: Device to run the model on (CPU or CUDA/MPS)
        """
        super().__init__()
        self.device = device
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.openai_clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).to(self.device)
        self.openai_clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1).to(self.device)
        self.resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_resized = self.resize(x)
        x_normalized = ((x_resized - self.openai_clip_mean) / self.openai_clip_std).contiguous()
        features = self.model.get_image_features(pixel_values=x_normalized)
        
        return features


class ResNet18Embedding(torch.nn.Module):
    
    def __init__(self, layer: str, device: str):
        """
        Initialize ResNet18 embedding model.
        
        Args:
            layer: Which layer to extract features
            device: Device to run the model on (CPU or CUDA/MPS)
        """
        super().__init__()
        self.device = device

        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            original_model = torchvision.models.resnet18(weights=weights)
        except:
            original_model = torchvision.models.resnet18(pretrained=True)
            
        original_model = original_model.to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(self.device)

        if layer == "last":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-2])
        else:
            raise ValueError("Invalid layer name")
            
        self.resize = transforms.Resize((224, 224), antialias=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract ResNet18 embeddings from input images.
        """
        x = self.resize(images)
        x = (x - self.mean) / self.std
        x = x.contiguous()
        x = self.features(x)
        
        out = torch.mean(x, dim=[2, 3])
        
        return out