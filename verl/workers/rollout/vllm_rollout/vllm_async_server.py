# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# ==================== vLLM Version Detection & Compatibility ====================
import vllm
VLLM_VERSION = tuple(map(int, getattr(vllm, '__version__', '0.0.0').split('.')[:2]))
print(f"Detected vLLM version: {VLLM_VERSION}")

# Back-fill symbols removed in vLLM 0.11+
import vllm.utils
if not hasattr(vllm.utils, "get_tcp_uri"):
    def get_tcp_uri(host: str, port: int) -> str:
        """Return a TCP URI that vLLM <0.11 used to expose."""
        return f"tcp://{host}:{port}"
    vllm.utils.get_tcp_uri = get_tcp_uri

# Import with fallback for moved modules
try:
    from vllm.utils import FlexibleArgumentParser, get_tcp_uri
except ImportError:
    from vllm.utils import FlexibleArgumentParser
    # get_tcp_uri is already backfilled above

# V1 engine imports - these are most likely to have changed
V1_AVAILABLE = True
try:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.engine.utils import CoreEngineProcManager
    from vllm.v1.executor.abstract import Executor
except (ImportError, ModuleNotFoundError) as e:
    print(f"V1 engine components not available: {e}, falling back to V0")
    V1_AVAILABLE = False
    try:
        # V0 compatibility imports
        from vllm.engine.async_llm_engine import AsyncLLMEngine as AsyncLLM
        from vllm.engine.llm_engine import LLMEngine
        Executor = None  # V0 doesn't have this abstraction
        CoreEngineProcManager = None
        EngineCoreProc = None
    except ImportError:
        print("Could not import vLLM V0 engine components either")
        raise

# SamplingMetadata import fallback
try:
    from vllm.model_executor.sampling_metadata import SamplingMetadata
except ImportError:
    try:
        from vllm.sampling_metadata import SamplingMetadata
    except ImportError:
        # Create dummy class if not found anywhere
        class SamplingMetadata:
            pass
        print("Warning: Could not import SamplingMetadata, using dummy class")

# WorkerWrapperBase import fallback
try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ImportError:
    try:
        from vllm.worker.base_worker import WorkerWrapperBase
    except ImportError:
        from vllm.worker.worker import WorkerWrapperBase

# =================================================================================

import argparse
import asyncio
import inspect
import json
import logging
import os
from pprint import pprint
from typing import Any, Callable, Optional

import cloudpickle as pickle
import numpy as np
import ray
import zmq
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_app

# init_app_state signature compatibility
try:
    from vllm.entrypoints.openai.api_server import init_app_state
except ImportError:
    # If init_app_state doesn't exist, create a dummy async function
    async def init_app_state(*args, **kwargs):
        print("Warning: init_app_state not found, skipping")
    INIT_APP_STATE_PARAMS = 0
else:
    # Determine signature at import time
    INIT_APP_STATE_PARAMS = len(inspect.signature(init_app_state).parameters)

from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches
from verl.workers.config import HFModelConfig, RewardModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address, run_unvicorn
from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class ExternalZeroMQDistributedExecutor:
    """An executor that engines are launched by external ray actors."""
    
    uses_ray: bool = False

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self._init_executor()

    def _init_executor(self) -> None:
        dp_rank_local = self.vllm_config.parallel_config.data_parallel_rank_local
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size

        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        addresses = addresses[dp_rank_local * tp_size : (dp_rank_local + 1) * tp_size]
        
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            if address.startswith("tcp://["):
                socket.setsockopt(zmq.IPV6, 1)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        **kwargs_extra: Any,
    ) -> list[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            if VLLM_VERSION >= (0, 11):
                sent_method = pickle.dumps(method)
            else:
                sent_method = method
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))

        for output in outputs:
            if isinstance(output, Exception):
                raise output
        return outputs

    def check_health(self):
        return


class vLLMHttpServerBase:
    """vLLM http server in single node."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        super().__init__()

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.config.max_model_len = self.config.prompt_length + self.config.response_length
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address)
            logger.info(
                f"vLLMHttpServer, replica_rank: {self.replica_rank}, master address: {self._master_address}, "
                f"master port: {self._master_port}, data parallel master port: {self._dp_master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

    def get_master_address(self):
        """Get master address and port for data parallel."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        
        if self.config.get("limit_images", None):
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        # Override default generation config
        override_generation_config = dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )
        logger.info(f"override_generation_config: {override_generation_config}")
        
        quantization = self.config.quantization
        fp8_block_quant_kwargs = None
        if quantization is not None:
            if quantization == "fp8":
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
                apply_vllm_fp8_patches()
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")

        # Build args dict
        args = {
            "model": self.model_config.local_path,
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            **engine_kwargs,
        }
        
        if fp8_block_quant_kwargs:
            args["hf_overrides"] = {"quantization_config": fp8_block_quant_kwargs}

        if self.config.prometheus.enable and self.config.prometheus.served_model_name:
            served_model_name = self.config.prometheus.served_model_name
            if "/" in served_model_name:
                served_model_name = served_model_name.split("/")[-1]
            args["served_model_name"] = served_model_name

        if self.config.expert_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size
            
            args.update({
                "enable_expert_parallel": self.config.expert_parallel_size > 1,
                "data_parallel_size": self.config.data_parallel_size,
                "data_parallel_size_local": data_parallel_size_local,
                "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                "data_parallel_address": self._master_address,
                "data_parallel_rpc_port": self._master_port,
            })

        # Update lora-related args
        if self.model_config.lora_rank > 0:
            args.update({
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank),
            })

        server_args = ["serve"]
        for k, v in args.items():
            if isinstance(v, bool):
                if v:
                    server_args.append(f"--{k}")
            elif v is not None:
                server_args.append(f"--{k}")
                server_args.append(json.dumps(v) if isinstance(v, dict) else str(v))

        if self.replica_rank == 0:
            pprint(server_args)

        # Parse arguments
        CMD_MODULES = [vllm.entrypoints.cli.serve]
        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        
        server_args_parsed = parser.parse_args(args=server_args)
        server_args_parsed.model = server_args_parsed.model_tag
        if server_args_parsed.subparser in cmds:
            cmds[server_args_parsed.subparser].validate(server_args_parsed)

        # 2. setup distributed executor backend
        if V1_AVAILABLE and len(self.workers) > 0:
            distributed_executor_backend = ExternalZeroMQDistributedExecutor
        else:
            distributed_executor_backend = None
            
        server_args_parsed.distributed_executor_backend = distributed_executor_backend

        zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in self.workers])
        logger.info(
            f"replica_rank={self.replica_rank}, node_rank={self.node_rank}, nnodes={self.nnodes}, "
            f"get worker zmq addresses: {zmq_addresses}"
        )
        os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        # 3. launch server
        if self.node_rank == 0:
            await self.run_server(server_args_parsed)
        else:
            await self.run_headless(server_args_parsed)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        
        # Create engine config with version compatibility
        if V1_AVAILABLE:
            try:
                vllm_config = engine_args.create_engine_config(usage_context=usage_context)
            except TypeError:
                # Older signature might not have usage_context
                vllm_config = engine_args.create_engine_config()
        else:
            vllm_config = engine_args.create_engine_config()

        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        # Create engine client with version compatibility
        if V1_AVAILABLE:
            try:
                engine_client = AsyncLLM.from_vllm_config(
                    vllm_config=vllm_config,
                    usage_context=usage_context,
                    disable_log_requests=engine_args.disable_log_requests,
                    disable_log_stats=engine_args.disable_log_stats,
                )
            except TypeError:
                # Try without usage_context
                engine_client = AsyncLLM.from_vllm_config(
                    vllm_config=vllm_config,
                    disable_log_requests=engine_args.disable_log_requests,
                    disable_log_stats=engine_args.disable_log_stats,
                )
        else:
            # V0 engine
            engine_client = AsyncLLM.from_engine_args(engine_args)

        # Don't keep the dummy data in memory
        if hasattr(engine_client, 'reset_mm_cache'):
            await engine_client.reset_mm_cache()

        app = build_app(args)
        
        # ------------- init_app_state signature compatibility -----------------
        if INIT_APP_STATE_PARAMS == 3:
            # old signature: (engine_client, vllm_config, app_state)
            await init_app_state(engine_client, vllm_config, app.state)
        elif INIT_APP_STATE_PARAMS == 4:
            # new signature: (engine_client, vllm_config, app_state, args)
            await init_app_state(engine_client, vllm_config, app.state, args)
        else:
            logger.warning(f"Unexpected init_app_state signature with {INIT_APP_STATE_PARAMS} params")
            # Try with minimal args
            await init_app_state(engine_client, vllm_config, app.state)
        # --------------------------------------------------------------------
        
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        if not V1_AVAILABLE:
            logger.error("Headless mode requires vLLM V1 engine")
            raise RuntimeError("V1 engine not available")

        # Create the EngineConfig.
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        
        try:
            vllm_config = engine_args.create_engine_config(usage_context=usage_context, headless=True)
        except TypeError:
            # Older signature might not have headless parameter
            vllm_config = engine_args.create_engine_config(usage_context=usage_context)

        parallel_config = vllm_config.parallel_config
        local_engine_count = parallel_config.data_parallel_size_local

        host = parallel_config.data_parallel_master_ip
        port = engine_args.data_parallel_rpc_port
        handshake_address = get_tcp_uri(host, port)

        # Create the engines.
        self.engine_manager = CoreEngineProcManager(
            target_fn=EngineCoreProc.run_engine_core,
            local_engine_count=local_engine_count,
            start_index=vllm_config.parallel_config.data_parallel_rank,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=False,
            handshake_address=handshake_address,
            executor_class=Executor.get_class(vllm_config),
            log_stats=not engine_args.disable_log_stats,
        )

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        max_tokens = self.config.max_model_len - len(prompt_ids)
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids, multi_modal_data={"image": image_data} if image_data else None
        )

        # Add lora request
        lora_request = None
        if self.model_config.lora_rank > 0:
            # Check if lora is already loaded
            try:
                loras = await self.engine.list_loras()
                lora_loaded = VLLM_LORA_INT_ID in loras
            except Exception:
                lora_loaded = False
                
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        generator = self.engine.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
        )

        # Get final response
        final_res: Optional[RequestOutput] = None
        try:
            async for output in generator:
                final_res = output
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise
            
        assert final_res is not None, "No output generated"

        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs is not None and final_res.outputs[0].logprobs:
            log_probs = []
            for i, logprob_dict in enumerate(final_res.outputs[0].logprobs):
                if logprob_dict and token_ids[i] in logprob_dict:
                    log_probs.append(logprob_dict[token_ids[i]].logprob)
                else:
                    log_probs.append(0.0)
                    
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            if self.node_rank == 0:
                try:
                    await self.engine.wake_up(tags=["kv_cache", "weights"])
                except Exception as e:
                    logger.warning(f"wake_up failed: {e}")
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            if self.node_rank == 0:
                try:
                    await self.engine.wait_for_requests_to_drain()
                    await self.engine.reset_prefix_cache()
                except Exception as e:
                    logger.warning(f"sleep prep failed: {e}")
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            if self.node_rank == 0:
                try:
                    await self.engine.reset_prefix_cache()
                    await self.engine.sleep(level=1)
                except Exception as e:
                    logger.warning(f"sleep failed: {e}")
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def wait_for_requests_to_drain(self):
        if hasattr(self.engine, 'wait_for_requests_to_drain'):
            try:
                await self.engine.wait_for_requests_to_drain()
            except Exception as e:
                logger.warning(f"wait_for_requests_to_drain failed: {e}")


@ray.remote(num_cpus=1)
class vLLMHttpServer(vLLMHttpServerBase):
    """vLLM http server in single node."""

    def __init__(
        self,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        super().__init__(config, model_config, rollout_mode, workers, replica_rank, node_rank, gpus_per_node, nnodes)


_rollout_worker_actor_cls = ray.remote(vLLMAsyncRollout)


class vLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = vLLMHttpServer

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # Get node_id of all workers
        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.workers
            ]
        )

        # For non-data parallel case, there's only one server
        nnodes, gpus_per_node = self.nnodes, self.gpus_per_node
        if self.config.data_parallel_size == 1:
            nnodes = 1
            gpus_per_node = self.world_size

        # Create server actor in each node with node affinity
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node]
            node_id = worker_node_ids[node_rank * gpus_per_node]
            name = (
                f"vllm_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"vllm_server_reward_{self.replica_rank}_{node_rank}"
            )
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            self.servers.append(server)

        # Launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # Get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Sleep each rollout server."""
        if hasattr(self.servers[0], 'wait_for_requests_to_drain'):
            try:
                await self.servers[0].wait_for_requests_to_drain.remote()
            except Exception as e:
                logger.warning(f"wait_for_requests_to_drain failed: {e}")
                
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])


def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens for Qwen2.5-VL."""
    if processor is not None and hasattr(processor, 'image_processor') and \
       "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)
        mask = np.ones(len(prompt_ids), dtype=bool)
        is_value = prompt_ids == processor.image_token_id
        mask[1:] &= ~(is_value[1:] & is_value[:-1])
        return prompt_ids[mask].tolist()
    return prompt_ids