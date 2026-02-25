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

class ImageLineTool(BaseTool):
    """A tool for drawing a straight line on an image based on coordinates.

    This tool provides a line-drawing functionality on an image,
    with rate limiting and concurrent execution support through Ray.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the line-drawing operation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "image_line_tool",
                "description": (
                    "Draws a straight line on a specific region of an image based on start and end coordinates."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": (
                                "The starting point of the line, as [x1, y1] coordinates."
                            ),
                        },
                        "end": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": (
                                "The ending point of the line, as [x2, y2] coordinates."
                            ),
                        }
                    },
                    "required": ["start", "end"],
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
        logger.info(f"Initialized ImageLineTool with config: {config}")

    def _save_image_debug_info(self, image_with_line, x1, y1, x2, y2, output_path):
        # Generate unique folder name
        instance_id = str(uuid4())[:8]
        instance_path = os.path.join(output_path, f"instance_{instance_id}")
        os.makedirs(instance_path, exist_ok=True)

        # Save the images
        image_with_line.save(os.path.join(instance_path, "original_with_line.jpg"))

        # Prepare metadata
        metadata = {
            "start": [x1, y1],
            "end": [x2, y2],
            "original_image_size": image_with_line.size
        }

        # Save metadata
        with open(os.path.join(instance_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved debug info to {instance_path}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, image: str | Image.Image, instance_id: Optional[str] = None, **kwargs) -> str:
        """
        Creates a new instance for image line-drawing tool.

        This method initializes a new session for an image, which can then be used
        for operations like drawing a line. It fetches the image from various sources
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
            expected_tool_calls: Optional list of expected tool calls for reward computation.
            coordinate_tolerance: Pixel tolerance for coordinate matching (default: 50).

        Returns:
            The unique identifier for the created instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        if 'bytes' in image:
            # convert raw bytes to PIL.Image
            image = Image.open(BytesIO(image['bytes']))

        img = fetch_image({"image": image})
        
        # Extract expected tool calls for reward computation
        expected_tool_calls = kwargs.get("expected_tool_calls", [])
        coordinate_tolerance = kwargs.get("coordinate_tolerance", 50.0)
        
        # Parse expected coordinates from tool calls
        expected_start = None
        expected_end = None
        if expected_tool_calls and len(expected_tool_calls) > 0:
            params = expected_tool_calls[0].get("parameters", {})
            expected_start = params.get("start")
            expected_end = params.get("end")
        
        self._instance_dict[instance_id] = {
            "image": img,
            "response": "",
            "reward": 0.0,
            "executed_calls": [],  # Track executed tool calls
            "expected_start": expected_start,
            "expected_end": expected_end,
            "coordinate_tolerance": coordinate_tolerance,
        }
        logger.info(f"Created instance {instance_id} with expected coords: start={expected_start}, end={expected_end}")
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[dict, float, dict]:
        start = parameters.get("start")
        end = parameters.get("end")

        if not start or not end:
            return (
                {"text": "Error: start and end parameters are required."},
                -0.1,  # Penalize malformed calls
                {"success": False},
            )

        instance_data = self._instance_dict[instance_id]
        image = instance_data["image"]
        image_width, image_height = image.size

        try:
            # Validate and clamp coordinates to image bounds
            x1 = max(0, min(float(start[0]), image_width))
            y1 = max(0, min(float(start[1]), image_height))
            x2 = max(0, min(float(end[0]), image_width))
            y2 = max(0, min(float(end[1]), image_height))

            # Check for zero-length line
            import math
            line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if line_length < 1.0:
                error_msg = f"Line is too short (length={line_length:.2f}). Start and end points are nearly identical."
                logger.warning(error_msg)
                return {"text": error_msg}, -0.05, {"success": False}

            image_with_line = image.copy()
            draw = ImageDraw.Draw(image_with_line)
            draw.line((x1, y1, x2, y2), fill='red', width=3)
            
            # Store executed coordinates for calc_reward
            executed_call = {
                "start": [x1, y1],
                "end": [x2, y2],
            }
            instance_data["executed_calls"].append(executed_call)
            
            # Save debug info (enabled for development/debugging)
            # Uncomment output_path to enable: output_path="/users/badralmahouri/scratch/run_outputs/debug_output"
            # self._save_image_debug_info(image_with_line, x1, y1, x2, y2, output_path=output_path)
            
        except Exception as e:
            logger.error(f"Error processing image line-drawing: {e}")
            return {"text": f"Error processing image line-drawing: {e}"}, -0.1, {"success": False}

        response_text = f"Drew a line on the image from ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})."
        
        # Compute step reward based on coordinate accuracy if expected values available
        step_reward = 0.0
        expected_start = instance_data.get("expected_start")
        expected_end = instance_data.get("expected_end")
        if expected_start and expected_end:
            step_reward = self._compute_coordinate_similarity(
                [x1, y1], [x2, y2], expected_start, expected_end,
                instance_data.get("coordinate_tolerance", 50.0)
            )
            logger.info(f"Execute step reward: {step_reward:.4f}")

        return (
            {
                "image": [image_with_line],
                "text": response_text,
            },
            step_reward,
            {"success": True, "executed_start": [x1, y1], "executed_end": [x2, y2]},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    def _compute_coordinate_similarity(
        self, 
        pred_start: list, 
        pred_end: list, 
        expected_start: list, 
        expected_end: list,
        tolerance: float = 50.0
    ) -> float:
        """
        Compute similarity between predicted and expected line endpoints.
        Returns a score in [0, 1] based on normalized distance.
        
        Handles line symmetry: A→B is equivalent to B→A.
        
        Args:
            pred_start: Predicted start coordinates [x, y]
            pred_end: Predicted end coordinates [x, y]
            expected_start: Expected start coordinates [x, y]
            expected_end: Expected end coordinates [x, y]
            tolerance: Pixel tolerance for normalization (default 50)
        
        Returns:
            Similarity score in [0, 1]
        """
        import math
        
        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Check both directions: A→B and B→A (lines are symmetric)
        # Direction 1: pred_start→expected_start, pred_end→expected_end
        start_dist_1 = distance(pred_start, expected_start)
        end_dist_1 = distance(pred_end, expected_end)
        avg_dist_1 = (start_dist_1 + end_dist_1) / 2.0
        
        # Direction 2: pred_start→expected_end, pred_end→expected_start (reversed)
        start_dist_2 = distance(pred_start, expected_end)
        end_dist_2 = distance(pred_end, expected_start)
        avg_dist_2 = (start_dist_2 + end_dist_2) / 2.0
        
        # Use the better match (smaller distance)
        avg_dist = min(avg_dist_1, avg_dist_2)
        
        # Normalize using exponential decay
        # Perfect match = 0 distance -> score 1.0
        # Large distance (>tolerance) -> score approaches 0
        similarity = math.exp(-avg_dist / tolerance)
        
        return similarity

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        Calculate final reward based on tool execution accuracy.
        
        This is called at the end of the rollout to compute the tool-level reward.
        Compares executed line coordinates against expected coordinates.
        
        Args:
            instance_id: The instance identifier
        
        Returns:
            Reward score in [0, 1] based on coordinate accuracy
        """
        if instance_id not in self._instance_dict:
            logger.warning(f"calc_reward called for unknown instance: {instance_id}")
            return 0.0
        
        instance_data = self._instance_dict[instance_id]
        executed_calls = instance_data.get("executed_calls", [])
        expected_start = instance_data.get("expected_start")
        expected_end = instance_data.get("expected_end")
        tolerance = instance_data.get("coordinate_tolerance", 50.0)
        
        # No tool calls executed -> zero reward
        if not executed_calls:
            logger.info(f"calc_reward: No executed calls for instance {instance_id}")
            return 0.0
        
        # No expected values -> can't compute reward
        if not expected_start or not expected_end:
            logger.info(f"calc_reward: No expected coordinates for instance {instance_id}")
            return 0.0
        
        # Use the first executed call for reward computation
        # (or could use the last one, or best one)
        executed = executed_calls[0]
        pred_start = executed["start"]
        pred_end = executed["end"]
        
        reward = self._compute_coordinate_similarity(
            pred_start, pred_end, expected_start, expected_end, tolerance
        )
        
        logger.info(
            f"calc_reward: instance={instance_id}, pred=({pred_start}, {pred_end}), "
            f"expected=({expected_start}, {expected_end}), reward={reward:.4f}"
        )
        
        return reward