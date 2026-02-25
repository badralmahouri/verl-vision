# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
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

import logging
import os
import json
import threading
from contextlib import ExitStack
from enum import Enum
from math import ceil, floor
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
from io import BytesIO

import ray
import ray.actor
from PIL import Image, ImageDraw
from qwen_vl_utils import fetch_image

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2

@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
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
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

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
                    # TODO we should make this available to the tool caller
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

class ImageFlipTool(BaseTool):
    """A tool for flipping an image horizontally or vertically.

    This tool provides a flipping functionality for images,
    with rate limiting and concurrent execution support through Ray.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the rotation operation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    MIN_DIMENSION = 28

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "image_flip_tool",
                "description": (
                    "Flip an image horizontally or vertically."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["horizontal", "vertical"],
                            "description": "The direction in which to flip the image."
                        },
                    },
                    "required": ["direction"],
                },
            }
        })
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
        logger.info(f"Initialized ImageFlipTool with config: {config}")

    # TODO: remove it afterwards
    def _save_image_debug_info(self, image, flipped_image, direction, output_path):
        # Generate unique folder name
        instance_id = str(uuid4())[:8]
        instance_path = os.path.join(output_path, f"instance_{instance_id}")
        os.makedirs(instance_path, exist_ok=True)

        # Save the images
        image.save(os.path.join(instance_path, "original_with_box.jpg"))
        flipped_image.save(os.path.join(instance_path, "flipped.jpg"))

        # Prepare metadata
        metadata = {
            "direction": direction
        }

        # Save metadata
        with open(os.path.join(instance_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved debug info to {instance_path}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, image: str | Image.Image, instance_id: Optional[str] = None, **kwargs) -> str:
        """
        Creates a new instance for image flip tool.

        This method initializes a new session for an image, which can then be used
        for operations like flipping. It fetches the image from various sources
        and stores it internally.

        Args:
            instance_id: An optional unique identifier for the instance. If not
                provided, a new UUID will be generated.
            image: image can be one of the following:
                - A PIL.Image.Image object.
                - A string containing an HTTP or HTTPS URL.
                - A string containing a local file path.
                - A string containing a file URI (e.g., "file:///path/to/image.jpg").
                - A string containing a base64-encoded image in the format of "data:image/jpeg;base64,..."

        Returns:
            The unique identifier for the created instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        if isinstance(image, dict) and 'bytes' in image:
            # convert raw bytes to PIL.Image
            image = Image.open(BytesIO(image['bytes']))

        img = fetch_image({"image": image})
        self._instance_dict[instance_id] = {
            "image": img,
            "response": "",
            "reward": 0.0,
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[dict, float, dict]:
        direction = parameters.get("direction")

        if direction not in ["horizontal", "vertical"]:
            error_msg = "Error: direction parameter is missing or invalid."
            logger.warning(f"Tool execution failed: {error_msg}")
            return {"text": error_msg}, -0.05, {"success": False}

        instance_data = self._instance_dict[instance_id]
        image = instance_data["image"]
        image_width, image_height = image.size

        try:
            if direction == "horizontal":
                flipped_image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
            else:  # vertical
                flipped_image = image.transpose(method=Image.FLIP_TOP_BOTTOM)

            # self._save_image_debug_info(image, flipped_image, direction, output_path="/users/jsaydali/scratch/run_outputs/image_outputs")
        except Exception as e:
            logger.error(f"Error processing image flip: {e}")
            return {"text": f"Error processing image flip: {e}"}, -0.05, {"success": False}

        response_text = f"Flipped the image {direction}ly."
        return (
            {
                "image": [flipped_image],
                "text": response_text,
            },
            0.0,
            {"success": True},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]