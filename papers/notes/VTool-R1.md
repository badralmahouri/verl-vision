# VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use
**Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, Klara Nahrstedt (UIUC, UMich). ICLR 2026.**

## Key Idea
VTool-R1 is the first RFT (reinforcement learning finetuning) framework that trains VLMs to generate multimodal chains of thought by interleaving text reasoning with intermediate visual reasoning steps produced by external Python-based image editing tools. The model learns *when and how* to use visual tools purely from outcome-based rewards (final answer correctness), with no process-level supervision of tool usage.

## Method

### Architecture and Rollout
- Two-stage iterative inference: (1) model generates a first response that may contain a Python tool call, (2) tool executes externally producing a modified image, (3) the modified image is fed back as a second image input alongside the original, (4) model generates final answer conditioned on both images.
- Single-turn tool use only (max one tool call per question). Multi-turn left for future work.
- The edited image is injected as a *new image input* to the model, not inserted inline into the token sequence. This is different from text-based tool use (e.g., Search-R1) where tool results are inserted into the response sequence.

### Training
- GRPO (Group Relative Policy Optimization) -- critic-free, group-based advantage estimation. Same as DeepSeek-R1.
- Only the **final response** y (after tool execution, second rollout) is optimized. The intermediate tool-invoking response y' is NOT directly optimized. This is a deliberate end-to-end design: the model learns that producing good tool calls leads to better final answers which leads to higher reward.
- Group size G=5 rollouts per input. KL coefficient beta=1e-2. Learning rate 1e-6 with AdamW.
- Global batch size 32 (tables) or 256 (charts). Temperature 1.0, bf16.
- Training infrastructure: **VeRL** (same framework we use). 8-16 H100 GPUs for 3B/7B, 8 H200 for 32B.

### Visual Editing Toolset
- Python functions for selective attention on charts/tables: highlight column/row, mask column/row, draw bounding box on column/row.
- Implemented with OpenCV. Tools take PIL images + bounding box dicts as input, return modified PIL images.
- Tool call success is defined as: Python code executes without exceptions AND a valid PIL image is returned.

### Prompt Design
- Highly structured prompt template with: system instructions, Python function signatures with docstrings, goal description, numbered requirements, and **4 in-context few-shot examples** showing thought-action-observation-answer format.
- Bounding box metadata (column/row coordinates) provided in the prompt as pre-computed dictionaries.
- The model outputs in a strict format: THOUGHT 0 -> ACTION 0 (python code block) -> OBSERVATION -> THOUGHT 1 -> ANSWER -> FINAL ANSWER -> TERMINATE.

### Reward Design
- **Outcome-based only**: reward = 1 if final answer matches ground truth (assessed by a lightweight LLM judge for open-ended answers, or exact string match for closed-ended).
- **No format reward** -- the structured prompt template is sufficient for format compliance.
- **Process-based rewards explicitly rejected**: penalizing failed tool calls causes the model to avoid tools entirely; rewarding successful tool calls causes reward hacking (model generates superficially valid tool calls that don't help reasoning). This is a critical finding.

## Results

### Main Numbers (Accuracy)
| Model | Chart (no tools) | Chart (VTool-R1) | Table (no tools) | Table (VTool-R1) |
|-------|------------------|-------------------|-------------------|-------------------|
| Qwen2.5-VL 3B | 51.8 | **64.0** (+12.2) | 41.3 | **57.9** (+16.6) |
| Qwen2.5-VL 7B | 76.2 | **80.7** (+4.5) | 64.7 | **71.7** (+7.0) |
| Qwen2.5-VL 32B | 88.0 | 86.7 (-1.3) | 86.2 | 84.5 (-1.7) |

- VTool-R1 7B matches or exceeds GPT-4o with tool use (80.7 vs 80.5 on charts, 71.7 vs 77.0 on tables).
- VTool-R1 significantly outperforms Deepeyes (concurrent work): 80.7 vs 60.0 on charts.
- **32B model does not benefit** -- it is already strong enough to answer directly without tools. Tool use actually slightly hurts.
- Untrained models prompted for tool use perform *worse* than direct inference (24.6 vs 51.8 for 3B on charts). Tools only help after RL training.

### Training Dynamics
- Accuracy converges within ~50 training steps (roughly one epoch).
- Tool call frequency is **non-monotonic**: models overuse tools early (due to prompt instruction exposure), then learn to be more selective. The model learns *when not to use tools*.
- Tool call success rate steadily increases on table tasks, fluctuates on chart tasks.
- 3B model becomes more cautious with tool use over time; 32B maintains higher tool use rate but also shows adaptive decline periods.

### Failure Modes
1. Correct tool call + wrong reasoning on the edited image (model misreads numbers).
2. Correct tool identification but bounding box physically obscures relevant data.
3. Model skips tool use when it should have used it.
4. Code execution fails (invalid Python).

## Relevant to Our Project

### Direct parallels
1. **Same framework (VeRL + GRPO)**: VTool-R1 uses VeRL for training infrastructure and GRPO for optimization, exactly what we use. Their training recipe is directly applicable.

2. **The "tool use makes untrained models worse" finding**: Their Table 1 shows that prompting untrained Qwen2.5-VL for tool use *drops* performance catastrophically (51.8 -> 24.6 for 3B). This is exactly our Apertus problem -- a model with no tool-calling prior performs worse when asked to use tools. VTool-R1 shows that RL can bridge this gap, but their starting point (Qwen) is much more capable than Apertus.

3. **Outcome-based reward is king, process reward is dangerous**: They explicitly tried and rejected process-based rewards for tool use. Penalizing bad tool calls -> model avoids tools entirely. Rewarding good tool calls -> reward hacking. This validates our approach of using final-answer correctness (IoU + answer match) as the reward signal. We should NOT add partial rewards for "valid JSON tool call" -- this is the kind of process reward they found causes hacking.

4. **Edited image as new input, not inline tokens**: VTool-R1 feeds the tool-processed image back as a second image input to the model, not by inserting result tokens inline. Our `tool_agent_loop.py` injects tool results as text tokens into the response sequence. Their approach may be cleaner for visual tools specifically, since the "observation" is an image, not text. For our bbox crop tool, the result is currently an image description injected as text -- worth considering whether feeding the cropped image back as a new image input would work better.

5. **Few-shot examples in the prompt are critical**: VTool-R1 uses 4 detailed in-context examples showing the full thought-action-observation-answer flow. For Apertus (zero tool-calling prior), this is even more important. Our `apertus_tool_chat_template.jinja2` should include explicit examples of the `<|tools_prefix|>...<|tools_suffix|>` format with complete bbox tool calls.

### Ideas for the reward desert problem
6. **They started with Qwen which has SOME tool-calling ability** -- even untrained Qwen can occasionally produce valid Python code. Apertus cannot even produce `<|tools_prefix|>` without being forced. This means our forced prefix curriculum is solving a harder problem than what VTool-R1 addresses. Their approach assumes the model can at least *attempt* tool calls (even badly) so RL has signal to work with.

7. **Their 50-step convergence suggests our curriculum phases might be too short**: They see convergence at ~50 steps. Our curriculum phases are 20 steps each. If it takes 50 steps just for RL to refine an already-present behavior, we likely need much longer phases (50-100 steps per phase) for Apertus which is starting from zero.

8. **The non-monotonic tool use pattern is expected**: If we get Apertus to start making tool calls, we should not panic if tool call frequency drops during training. The model may be learning to be selective. What matters is final answer accuracy.

9. **Single-turn tool use is sufficient**: VTool-R1 only allows one tool call per question and achieves strong results. Our setup also uses single tool call (image_bbox_tool). This is a good design choice for a model with no prior -- multi-turn would make the exploration problem exponentially harder.

### What they did NOT solve (and we still need to)
10. **No curriculum or scaffolding for models with zero tool prior**: VTool-R1 assumes the model can at least attempt tool calls from the start (Qwen was instruction-tuned). They never address how to bootstrap tool use in a model that has never seen tool-calling syntax. Our forced prefix curriculum is a novel contribution that goes beyond VTool-R1's approach.

11. **No special token handling issues**: Qwen uses regular text tokens for tool calls (`<tool_call>` etc.), so `skip_special_tokens` is irrelevant. They never had our bug #7 where special tokens get stripped by the reward decoder. This confirms our `skip_special_tokens=False` fix is specific to Apertus's tokenizer design.

## Limitations / Open Questions

- **Single-turn only**: Limited to one tool call per question. Multi-turn tool use remains future work. They acknowledge this requires stronger VLMs.
- **Narrow task domain**: Only evaluated on structured image reasoning (charts + tables). No evidence this transfers to other visual reasoning tasks.
- **32B model regression**: Tool use slightly hurts the 32B model, suggesting diminishing returns for already-capable models. No analysis of when tool use helps vs hurts.
- **No ablation on the in-context examples**: Unclear how much of the tool-calling ability comes from the 4 few-shot examples vs the RL training. Would the model still learn to use tools with fewer or no examples?
- **Tool correctness is only approximate**: They use a proxy metric (no Python exceptions + valid PIL image returned) since there's no oracle verifier for tool call correctness.
- **LLM-based reward judge**: Using an LLM to assess answer correctness is not truly rule-based. Could introduce noise or bias. They call it "pseudo rule-based."
- **No analysis of exploration**: They don't discuss how GRPO handles the exploration problem -- what fraction of rollouts produce valid tool calls early in training? This is exactly our "reward desert" question: if most rollouts produce garbage tool calls, GRPO has no positive reward signal to learn from.
