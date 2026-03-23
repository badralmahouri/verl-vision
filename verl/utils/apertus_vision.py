"""Apertus discrete image tokenization via IBQ vision tokenizer.

Converts PIL images into discrete token strings that can be inserted directly
into text prompts. Uses the HF checkpoint token naming convention (not Emu3.5).

Token mapping (HF checkpoint):
  <|img_start|>      - token 131073 - marks image beginning
  <|img_end|>        - token 131074 - marks image end
  <|img_token_start|> - token 131075 - marks start of visual tokens
  <|img_end_of_row|>  - token 131076 - separates rows of visual tokens
  <|image|>          - token 131079 - placeholder in chat template
  <|visual token N|>  - tokens 131272+ - IBQ codebook entries (131072 total)

Usage:
    vq_model = load_vision_tokenizer("/path/to/Emu3.5-VisionTokenizer")
    image_str = encode_image(pil_image, vq_model)
    prompt = raw_text.replace("<|image|>", image_str, 1)
"""

import os

import numpy as np
import torch
from PIL import Image

# Token names matching the HF checkpoint (NOT Emu3.5 naming)
IMG_START = "<|img_start|>"
IMG_END = "<|img_end|>"
IMG_TOKEN_START = "<|img_token_start|>"
IMG_END_OF_ROW = "<|img_end_of_row|>"


def smart_resize(image: Image.Image, area: int = 512 * 512, ds_factor: int = 16) -> Image.Image:
    """Resize image to target area while maintaining aspect ratio, rounded to ds_factor.

    The IBQ encoder requires spatial dimensions divisible by 16 (due to 16x downsampling).
    This function finds the closest valid size that preserves aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height
    new_height = int((area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    new_height = ((new_height + ds_factor // 2) // ds_factor) * ds_factor
    new_width = ((new_width + ds_factor // 2) // ds_factor) * ds_factor
    return image.resize((new_width, new_height), Image.BICUBIC)


def format_image_string(ibq_tokens: torch.Tensor) -> str:
    """Format IBQ token grid into Apertus image token string.

    Output format:
        <|img_start|>H*W<|img_token_start|><|visual token 0|>...<|img_end_of_row|>...<|img_end|>

    where H and W are the token grid dimensions (image_height/16 x image_width/16).
    """
    h, w = ibq_tokens.shape
    rows = []
    for _h in range(h):
        row = "".join(f"<|visual token {int(ibq_tokens[_h, _w])}|>" for _w in range(w))
        rows.append(row)
    token_str = IMG_END_OF_ROW.join(rows)
    return f"{IMG_START}{h}*{w}{IMG_TOKEN_START}{token_str}{IMG_END}"


@torch.no_grad()
def encode_image(image: Image.Image, vq_model, image_area: int = 512 * 512) -> str:
    """Convert PIL image to Apertus image token string via IBQ.

    Pipeline: PIL Image -> resize -> normalize -> IBQ encode -> format token string

    Args:
        image: PIL Image (will be converted to RGB if needed)
        vq_model: IBQ vision tokenizer model (from load_vision_tokenizer)
        image_area: target image area in pixels (default 512x512 = 262144)

    Returns:
        Formatted image token string for insertion into prompt text.
    """
    image = image.convert("RGB")
    image = smart_resize(image, image_area)
    w, h = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image_tensor = torch.tensor(np.array(image) / 127.5 - 1.0).to(device, dtype).permute(2, 0, 1)
    _, _, token = vq_model.encode(image_tensor[None])
    token = token[-1].view(h // 16, w // 16)
    return format_image_string(token)


def load_vision_tokenizer(model_path: str = None, device: str = "cpu"):
    """Load IBQ vision tokenizer model.

    Args:
        model_path: Path to vision tokenizer. If None, reads from VQ_MODEL_PATH env var.
                    Supports local path (with config.yaml + model.ckpt) or HuggingFace model ID.
        device: Device to load model on. Default "cpu" to avoid GPU memory pressure.

    Returns:
        IBQ vision tokenizer model ready for encode_image().
    """
    if model_path is None:
        model_path = os.environ.get("VQ_MODEL_PATH")
    if model_path is None:
        raise ValueError(
            "No vision tokenizer path provided. Set VQ_MODEL_PATH env var "
            "or pass model_path argument."
        )

    # Use Emu3.5's build_vision_tokenizer if available (on PYTHONPATH)
    try:
        from vision_tokenizer import build_vision_tokenizer

        return build_vision_tokenizer("ibq", model_path, device=device)
    except ImportError:
        pass

    # Fallback: load directly via HuggingFace AutoModel
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model
