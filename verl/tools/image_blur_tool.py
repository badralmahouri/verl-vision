# Copyright 2023-2025 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Image Visual Tools - blur and unblur utilities.

This module contains tools for applying blur and unblur (sharpen)
operations. It defines a shared implementation and exposes two public
classes:

- `ImageBlurTool`: exposes blur operations (`gaussian`, `box`, `motion`).
- `ImageUnblurTool`: exposes sharpening operations (`sharpen`, `unsharp_mask`, `edge_enhance`).

Both tools share the same instance management and image-processing helpers.
"""

import logging
import os
import json
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
from io import BytesIO

import ray
import ray.actor
from PIL import Image, ImageFilter
from qwen_vl_utils import fetch_image

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    """Execution pool mode enumeration."""
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class VisualExecutionWorker:
    """Worker for executing visual processing operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="blur-rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing visual processing: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_visual_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize visual execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisualExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class _ImageVisualTool(BaseTool):
    """
    Shared implementation for image blur/unblur tools.
    """

    MIN_DIMENSION = 28

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize the ImageBlurTool.
        
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_visual_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        logger.info(f"Initialized ImageBlurTool with config: {config}")

    def _save_image_debug_info(self, image, blurred_image, blur_type, intensity, output_path):
        """Save debug information for troubleshooting."""
        instance_id = str(uuid4())[:8]
        instance_path = os.path.join(output_path, f"blur_instance_{instance_id}")
        os.makedirs(instance_path, exist_ok=True)

        image.save(os.path.join(instance_path, "original.jpg"))
        blurred_image.save(os.path.join(instance_path, "blurred.jpg"))

        metadata = {
            "blur_type": blur_type,
            "intensity": intensity
        }

        with open(os.path.join(instance_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved debug info to {instance_path}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, image: str | Image.Image, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a new instance for image blur tool.

        This method initializes a new session for an image, which can then be used
        for blur operations.

        """
        if instance_id is None:
            instance_id = str(uuid4())

        if isinstance(image, dict) and 'bytes' in image:
            image = Image.open(BytesIO(image['bytes']))

        img = fetch_image({"image": image})
        self._instance_dict[instance_id] = {
            "image": img,
            "original_image": img.copy(),  # Keep original for potential unblur
            "response": "",
            "reward": 0.0,
        }
        return instance_id

    def _apply_gaussian_blur(self, image: Image.Image, intensity: int) -> Image.Image:
        """Apply Gaussian blur to the image.
        
        """
        # Map intensity 1-10 to radius 1-20
        radius = intensity * 2
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _apply_box_blur(self, image: Image.Image, intensity: int) -> Image.Image:
        """Apply box blur to the image.
        
        """
        radius = intensity * 2
        return image.filter(ImageFilter.BoxBlur(radius=radius))

    def _apply_motion_blur(self, image: Image.Image, intensity: int) -> Image.Image:
        """Apply motion blur effect to the image.
        
        For simplicity, we simulate motion blur using multiple box blurs
        in the horizontal direction.

        """
        import numpy as np
        from PIL import ImageFilter
        
        # Create a simple motion blur kernel (horizontal)
        kernel_size = intensity * 3 + 2  # 5 to 32
        
        # Use a simple horizontal motion blur approximation
        # by applying directional blur multiple times
        result = image
        for _ in range(intensity):
            result = result.filter(ImageFilter.Kernel(
                size=(3, 3),
                kernel=[0, 0, 0, 1, 1, 1, 0, 0, 0],
                scale=3,
                offset=0
            ))
        
        return result

    def _apply_sharpen(self, image: Image.Image, intensity: int) -> Image.Image:
        """Apply sharpening filter to the image (opposite of blur).
        
        """
        result = image
        # Apply sharpen filter multiple times based on intensity
        for _ in range(intensity):
            result = result.filter(ImageFilter.SHARPEN)
        return result

    def _apply_unsharp_mask(self, image: Image.Image, intensity: int) -> Image.Image:
        """Apply unsharp mask filter to enhance edges and reduce blur.
        
        """
        radius = 2
        percent = intensity * 30  # 30 to 300
        threshold = 3
        return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    def _apply_edge_enhance(self, image: Image.Image, intensity: int) -> Image.Image:
        """Apply edge enhancement filter.

        """
        result = image
        for _ in range(min(intensity, 5)):  # Cap at 5 to avoid over-enhancement
            if intensity > 5:
                result = result.filter(ImageFilter.EDGE_ENHANCE_MORE)
            else:
                result = result.filter(ImageFilter.EDGE_ENHANCE)
        return result

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[dict, float, dict]:
        """Execute the blur operation on the image.
    
        """
        blur_type = parameters.get("blur_type")
        intensity = parameters.get("intensity")

        # Determine allowed operations from subclass
        valid_blur_types = getattr(self, "allowed_ops", None)
        if valid_blur_types is None:
            # Fallback to allowing all implemented ops
            valid_blur_types = ["gaussian", "box", "motion", "sharpen", "unsharp_mask", "edge_enhance"]

        if blur_type not in valid_blur_types:
            error_msg = f"Error: blur_type must be one of {valid_blur_types}, got '{blur_type}'"
            logger.warning(f"Tool execution failed: {error_msg}")
            return {"text": error_msg}, -0.05, {"success": False}

        # Validate intensity
        try:
            intensity = int(intensity)
            if not 1 <= intensity <= 10:
                raise ValueError("Intensity out of range")
        except (TypeError, ValueError):
            error_msg = f"Error: intensity must be an integer between 1 and 10, got '{intensity}'"
            logger.warning(f"Tool execution failed: {error_msg}")
            return {"text": error_msg}, -0.05, {"success": False}

        if instance_id not in self._instance_dict:
            error_msg = f"Error: instance_id '{instance_id}' not found"
            logger.warning(f"Tool execution failed: {error_msg}")
            return {"text": error_msg}, -0.05, {"success": False}

        instance_data = self._instance_dict[instance_id]
        image = instance_data["image"]

        try:
            if blur_type == "gaussian":
                processed_image = self._apply_gaussian_blur(image, intensity)
            elif blur_type == "box":
                processed_image = self._apply_box_blur(image, intensity)
            elif blur_type == "motion":
                processed_image = self._apply_motion_blur(image, intensity)
            elif blur_type == "sharpen":
                processed_image = self._apply_sharpen(image, intensity)
            elif blur_type == "unsharp_mask":
                processed_image = self._apply_unsharp_mask(image, intensity)
            elif blur_type == "edge_enhance":
                processed_image = self._apply_edge_enhance(image, intensity)
            else:
                raise ValueError(f"Unknown blur type: {blur_type}")

            # Update the stored image
            self._instance_dict[instance_id]["image"] = processed_image

        except Exception as e:
            logger.error(f"Error processing image blur: {e}")
            return {"text": f"Error processing image blur: {e}"}, -0.05, {"success": False}

        blur_ops = {"gaussian", "box", "motion"}
        if set(getattr(self, "allowed_ops", [])).intersection(blur_ops):
            action = "blur"
        else:
            action = "unblur"
        response_text = f"Applied {blur_type} ({action}) with intensity {intensity} to the image."
        return (
            {
                "image": [processed_image],
                "text": response_text,
            },
            0.0,
            {"success": True, "blur_type": blur_type, "intensity": intensity},
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward for this instance.
        
        """
        if instance_id in self._instance_dict:
            return self._instance_dict[instance_id].get("reward", 0.0)
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release resources for this instance.

        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class ImageBlurTool(_ImageVisualTool):
    """Tool exposing only blur operations: gaussian, box, motion."""

    allowed_ops = ["gaussian", "box", "motion"]


class ImageUnblurTool(_ImageVisualTool):
    """Tool exposing only unblur/sharpen operations: sharpen, unsharp_mask, edge_enhance."""

    allowed_ops = ["sharpen", "unsharp_mask", "edge_enhance"]
