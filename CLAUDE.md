# CLAUDE.md - Project Context for Claude Code

## What This Project Is

verl-vision is a fork of [verl](https://github.com/volcengine/verl) adapted for vision-language model RL training using GRPO (Group Relative Policy Optimization). We train VLMs to use tools (like `image_bbox_tool`) via multi-turn rollouts: the model generates a tool call, the tool executes and returns results, the model generates a final answer.

Two models are being trained:
- **Qwen2.5-VL-3B** — works well, achieves ~0.65 reward. Uses Hermes tool format (`<tool_call>...</tool_call>`). Serves as the working baseline.
- **Apertus 9.13B** (`apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss`) — the main challenge. Uses Apertus tool format (`<|tools_prefix|>...<|tools_suffix|>`). Has **zero tool-calling prior** — it was SFT'd on image-text only. All changes for Apertus MUST NOT break Qwen.

**Task**: The model sees an image + question ("what shape is at [bbox]?"), must call `image_bbox_tool` with bounding box coordinates to crop the region, then answer in `<boxed>answer</boxed>` format.

**Infrastructure**: Runs on CSCS GH200 nodes (4 GPUs, 98GB each). Jobs submitted via SLURM. SGLang 0.5.3 for inference. Ray for distributed training.

## Key Architecture

```
User prompt + image
    -> Agent Loop (tool_agent_loop.py) — state machine: PENDING -> GENERATING -> PROCESSING_TOOLS -> GENERATING -> TERMINATED
        -> SGLang generates response
        -> Tool parser extracts tool calls from token IDs
        -> Tool executes, result injected back
        -> Model generates final answer
    -> Reward function (reward_bbox.py) scores the full response
    -> GRPO updates model weights
```

The `response_mask` tracks which tokens get gradient: `1` = model-generated, `0` = injected (tool responses, forced prefix). Both types appear in `response_ids` passed to the reward function.

## Critical Files

### Core pipeline (modified for Apertus)
| File | Role |
|------|------|
| `verl/experimental/agent_loop/tool_agent_loop.py` | Agent loop state machine. Apertus stop tokens, forced prefix curriculum, tool parser fix |
| `verl/experimental/agent_loop/agent_loop.py` | Base agent loop. Passes `_global_step` to sampling_params |
| `verl/experimental/agent_loop/tool_parser.py` | Extracts tool calls from token IDs. `ApertusToolParser` at line 164. Uses `skip_special_tokens=False` (correct) |
| `verl/experimental/reward/reward_loop/naive.py` | Decodes response_ids -> text -> passes to reward function. **Line 55: skip_special_tokens** |
| `verl/utils/reward_score/reward_bbox.py` | Reward function. Regex-matches tool calls, computes IoU, scores answer |
| `verl/workers/rollout/sglang_rollout/async_sglang_server.py` | SGLang interface. Strips internal `_global_step` key |

### Config and deployment
| File | Role |
|------|------|
| `examples/sglang_multiturn/config/emu3_bbox_grpo.yaml` | Apertus GRPO training config |
| `examples/sglang_multiturn/config/bbox_grpo.yaml` | Qwen GRPO training config |
| `examples/sglang_multiturn/config/apertus_tool_chat_template.jinja2` | Custom Jinja2 chat template with explicit tool-call instructions |
| `examples/sglang_multiturn/config/bbox_only_tool_config.yaml` | Tool definition for `image_bbox_tool` |
| `slurm/apertus_bbox.slurm` | Apertus SLURM job script |
| `slurm/run_image_bbox_tool_example.slurm` | Qwen SLURM job script |
| `data/generate_random_bbox.py` | Generates training/val parquet data |
| `data_postprocess/verl_metrics.py` | Training entrypoint (wraps `verl.trainer.main_ppo`) |

### Apertus-specific infrastructure
| File | Role |
|------|------|
| `verl/utils/apertus_vision.py` | IBQ vision tokenizer: encodes images as discrete text tokens |
| Emu3.5 source at `/users/$USER/Emu3.5/src` | Required for `vision_tokenizer` module (PYTHONPATH) |
| VQ model at `/capstor/.../Emu3.5-VisionTokenizer` | IBQ codebook for image encoding |
| Checkpoint at `/capstor/.../apertus-8b-img-SFT-.../HF` | Apertus model weights |

## Problems Encountered and Solutions

### 1. Wrong stop tokens (first Apertus run)
**Symptom**: Every response maxed at 1024 tokens, 100% clip ratio, ~600s/step, zero tool calls.
**Root cause**: SGLang used `</s>` (id=2) as default stop but Apertus uses `<|assistant_end|>` (id=68).
**Fix**: Added stop_token_ids `[68, 72]` for Apertus format in `tool_agent_loop.py:150-157`.

### 2. SGLang OOM on Apertus 8B
**Symptom**: SGLang crashed during init.
**Fix**: `gpu_memory_utilization: 0.85` in config + `param_offload: True` for FSDP.

### 3. Qwen VL template forced on Apertus
**Symptom**: Template errors, wrong image handling.
**Root cause**: Apertus encodes images as discrete text tokens (IBQ), not as pixel embeddings like Qwen.
**Fix**: Separate image handling path in `tool_agent_loop.py:163-175`. Custom chat template at `apertus_tool_chat_template.jinja2`.

### 4. Logit bias +5.0 had no effect (job 1753116)
**Symptom**: 50 steps, zero tool calls. Model always outputs `<boxed>answer</boxed>` directly.
**Investigation**: Added debug logging, confirmed logit_bias reaches SGLang. Increased to +20.0.

### 5. Logit bias +20.0 causes degenerate repetition (job 1758332)
**Symptom**: Model outputs `<|tools_prefix|><|tools_prefix|><|tools_prefix|>...` endlessly.
**Conclusion**: **Model has zero tool-calling prior.** It can be forced to emit the token but has no knowledge of what comes after. No amount of logit bias can bridge this gap.
**Solution**: Switched to forced prefix approach.

### 6. Forced prefix: tool parser can't see prefix tokens (job 1759244)
**Symptom**: Model correctly completes bbox coordinates after forced prefix, but tool parser reports 0 tool calls.
**Root cause**: Forced prefix is in `prompt_ids`, but `extract_tool_calls()` only sees `agent_data.response_ids` (model-generated tokens). The `<|tools_prefix|>` that starts the tool call is missing.
**Fix**: Store `self._apertus_forced_prefix_ids` and prepend to parse_ids before extraction. `tool_agent_loop.py:343-345`.

### 7. Reward function never sees tool tokens (FIXED, NOT YET TESTED)
**Symptom**: Score is always 0 even when tool parser correctly detects tool calls.
**Root cause**: `verl/experimental/reward/reward_loop/naive.py:55` decodes with `skip_special_tokens=True`. This strips `<|tools_prefix|>` (id=71) and `<|tools_suffix|>` (id=72) — they're special tokens in Apertus' tokenizer. The reward regex `r'<\|tools_prefix\|>(.*?)<\|tools_suffix\|>'` in `reward_bbox.py:60` never matches.
**Fix**: Changed to `skip_special_tokens=False` in `naive.py:55`. Safe for Qwen because `<tool_call>` tags are regular text tokens. The tool parser (`tool_parser.py:137,191`) already correctly uses `skip_special_tokens=False`.

### 8. Reward debug output lost to Ray workers
**Symptom**: Per-sample reward details (`response_preview`, `score`, `iou`, `pred_bbox`) written to `logger.error()` go to Ray worker stderr and vanish when the job ends.
**Fix**: Added persistent JSONL file logger in `reward_bbox.py`. Writes to `$REWARD_DEBUG_LOG` (set in SLURM script to `slurm/logs/reward_debug_${SLURM_JOB_ID}.jsonl`). Propagated to Ray workers via `RAY_worker_env_REWARD_DEBUG_LOG`.

## Current Problem: The "Reward Desert"

Apertus has no tool-calling prior. RL (GRPO) can only reinforce behaviors the model already explores. If the model never produces a valid tool call, there's no positive reward to learn from.

### Approach: Forced Prefix Curriculum
Implemented in `tool_agent_loop.py:246-305`. Appends tool-call structure tokens to `prompt_ids`:

- **Steps 0-19**: Full prefix `<|tools_prefix|>[{"image_bbox_tool": {"bbox_2d": [` — model only completes the 4 bbox coordinates + closing syntax
- **Steps 20-39**: Partial `<|tools_prefix|>[{"image_bbox_tool": ` — model completes JSON body
- **Steps 40-59**: Minimal `<|tools_prefix|>` — model generates full tool call after the token
- **Steps 60+**: No prefix — model on its own

Prefix tokens get `response_mask=0` (no gradient). Global step passed from trainer via `_global_step` in sampling_params (`agent_loop.py` -> `tool_agent_loop.py:160`).

**Status**: Code is complete. The blocking reward decode bug (problem #7) has been fixed but **not yet tested in a training run**. Next step: submit a job and check `reward_debug_<JOBID>.jsonl`.

### Alternative ideas (not yet tried)
1. **SFT warm-up**: Fine-tune on 50-100 tool-calling examples before RL
2. **Longer curriculum phases**: 50-100 steps per phase instead of 20
3. **Permanent prefix first**: No curriculum decay — verify rewards work before adding decay
4. **Adaptive curriculum**: Decay based on reward signal, not fixed step count
5. **Partial JSON reward**: Small reward for valid JSON even with bad bbox

## Sources of Feedback (for autonomous debugging)

### 1. Persistent reward debug log (GROUND TRUTH)
**File**: `slurm/logs/reward_debug_<JOBID>.jsonl`
**Contains**: One JSON line per sample with:
- `response_preview` — first 200 chars of decoded response (shows if `<|tools_prefix|>` is present)
- `num_tool_calls` — whether the reward regex matched
- `score` — exact reward value
- `iou` — bounding box accuracy
- `pred_bbox` / `expected_bbox` — predicted vs ground truth coordinates
- `answer_score` — whether `<boxed>answer</boxed>` was correct
- `freetext_bbox` — whether bbox was found in free text (not tool call)

**How to use**: `cat slurm/logs/reward_debug_<JOBID>.jsonl | python -m json.tool` or `jq . < file.jsonl | head -50`
**Set by**: `REWARD_DEBUG_LOG` env var in `slurm/apertus_bbox.slurm`, propagated via `RAY_worker_env_REWARD_DEBUG_LOG`

### 2. SLURM metrics log
**File**: `slurm/logs/apertus_bbox_<JOBID>.log`
**Contains**: Per-step aggregated metrics extracted from SLURM stdout. Key metrics:
- `critic/score/mean` — average reward across batch
- `critic/score/max` — best reward in batch (any positive = some tool calls working)
- `response_length/mean` — should increase when forced prefix is active (~15+ tokens for prefix + completion)
- `response_length/clip_ratio` — 1.0 means all responses hit max length (bad — model isn't stopping)
- `num_turns/mean` — >2 means tool calls detected and executed. 2 = no tool calls.
- `timing_s/agent_loop/tool_calls/mean` — >0 means tool execution happened
- `actor/entropy` — should decrease as model learns a specific pattern
- `actor/pg_loss` — policy gradient loss

**Generated by**: Post-processing in `slurm/apertus_bbox.slurm` (sed + grep on SLURM output)

### 3. TensorBoard
**Directory**: `tensorboard_log/apertus_bbox_training/apertus_bbox_tool/`
**Contains**: Same scalar metrics as SLURM log + text traces (when `log_val_generations > 0` in config)
**Text traces**: Show actual decoded validation responses — can see tool call structure
**How to check**: `ls tensorboard_log/apertus_bbox_training/apertus_bbox_tool/events.out.tfevents*` for file dates

### 4. Auto-generated plots
**Directory**: `slurm/plots/apertus_bbox_<JOBID>/`
**Contains**: PDF plots of rewards, response length, advantages, KL/entropy, training dynamics
**Generated by**: `slurm/plot.py` called at end of SLURM script
**How to view**: `Read` the PDF files directly (Claude Code can read PDFs)

### 5. SLURM job status
**Command**: `squeue -u $USER` — check running/pending jobs
**Command**: `sacct -j <JOBID> --format=State,ExitCode,Elapsed` — check completed job status

### 6. Git diff and history
**Command**: `git diff` — see all uncommitted changes
**Command**: `git log --oneline -15` — recent commits on `feat/apertus-bbox-tool-support` branch

### 7. Live SLURM output (while job runs)
**File**: SLURM writes to `<jobname>_<jobid>.out` and `<jobname>_<jobid>.err` in the submit directory
**Useful for**: Watching training progress in real-time, catching crashes early

## Apertus Tokenizer Notes

- **Text tokenizer**: Bundled in the Apertus checkpoint (Mistral-inherited, vocab size 131,272)
- **Two sets of tool tokens**: Mistral-inherited (IDs 5-9) and Apertus-native (IDs 71-72). The built-in template uses Apertus format.
- `<|tools_prefix|>` = token ID 71 (SPECIAL token — stripped by `skip_special_tokens=True`)
- `<|tools_suffix|>` = token ID 72 (SPECIAL token — stripped by `skip_special_tokens=True`)
- `<|assistant_end|>` = token ID 68 (stop token for generation)
- `<tool_call>` in Qwen = regular text tokens (NOT special — preserved by `skip_special_tokens=True`)
- **Vision tokenizer**: Emu3.5 IBQ tokenizer encodes images as discrete text tokens (131,072 codebook). Text vocab 131,272 + vision 131,072 = total 262,400.

## Conventions

- **Commit messages**: Simple, clear, no AI attribution. Reference the specific bug/fix.
- **SLURM log naming**: `slurm/logs/<model>_bbox_<JOBID>.log`
- **Plot naming**: `slurm/plots/<model>_bbox_<JOBID>/`
- **Apertus-specific code**: Always gated by `self.tool_parser_name == "apertus"` or `tool_parser_name == "apertus"` to avoid breaking Qwen.
- **Config files**: `emu3_bbox_grpo.yaml` = Apertus, `bbox_grpo.yaml` = Qwen
