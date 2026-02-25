"""Vision tokenization utilities for pretokenized VLM training."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
from PIL import Image


class VisionTokenizer(ABC):
    """Abstract base for vision tokenization (continuous or discrete)."""
    
    token_type: str  # "continuous" or "discrete"
    
    @abstractmethod
    def encode(self, image: Image.Image) -> Dict[str, Any]:
        """Encode image to cacheable format."""
        pass
    
    @abstractmethod
    def get_input_embeds(self, cached: Dict[str, Any], text_embeds: torch.Tensor, 
                         image_positions: torch.Tensor) -> torch.Tensor:
        """Insert vision data into text embeddings."""
        pass


class QwenVLTokenizer(VisionTokenizer):
    """Continuous embeddings for Qwen2.5-VL."""
    
    token_type = "continuous"
    
    def __init__(self, model, processor):
        self.visual = model.visual
        self.processor = processor
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
    
    def encode(self, image: Image.Image) -> Dict[str, Any]:
        inputs = self.processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        grid_thw = inputs["image_grid_thw"].to(self.device)
        
        with torch.no_grad():
            embeds = self.visual(pixel_values, grid_thw=grid_thw)
        
        return {
            "type": "continuous",
            "embeddings": embeds.cpu().half().numpy(),
            "grid_thw": grid_thw.cpu().numpy(),
        }
    
    def get_input_embeds(self, cached: Dict[str, Any], text_embeds: torch.Tensor,
                         image_mask: torch.Tensor) -> torch.Tensor:
        embeds = torch.from_numpy(cached["embeddings"]).to(text_embeds.device, text_embeds.dtype)
        return text_embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(text_embeds), embeds)


class Emu3Tokenizer(VisionTokenizer):
    """Discrete tokens for Emu3."""
    
    token_type = "discrete"
    
    def __init__(self, image_tokenizer, image_processor):
        self.tokenizer = image_tokenizer
        self.processor = image_processor
        self.device = next(image_tokenizer.parameters()).device
    
    def encode(self, image: Image.Image) -> Dict[str, Any]:
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            codes = self.tokenizer.encode(pixel_values)
        
        return {
            "type": "discrete",
            "token_ids": codes.cpu().numpy(),
            "shape": list(codes.shape),
        }
    
    def get_input_embeds(self, cached: Dict[str, Any], text_embeds: torch.Tensor,
                         image_mask: torch.Tensor) -> torch.Tensor:
        # For discrete tokens, they're already in input_ids - no special handling needed
        return text_embeds


def create_vision_tokenizer(model_name: str, model=None, processor=None) -> Optional[VisionTokenizer]:
    """Factory function to create appropriate vision tokenizer."""
    if model is None:
        return None
    
    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower and "vl" in model_name_lower:
        return QwenVLTokenizer(model, processor)
    elif "emu3" in model_name_lower:
        from transformers import AutoModel, AutoImageProcessor
        img_tokenizer = AutoModel.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True)
        img_processor = AutoImageProcessor.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True)
        return Emu3Tokenizer(img_tokenizer.to(model.device), img_processor)
    
    return None
