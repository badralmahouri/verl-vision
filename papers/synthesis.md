# Paper Synthesis

Cross-cutting themes, contradictions, and concrete ideas for verl-vision drawn from the papers in this folder.

*Updated as new papers are read. Each section references the source paper notes in `notes/`.*

---

## Themes

### 1. Cold-start SFT before RL is not optional
Every paper that attempted RL-from-scratch on tool use or complex reasoning **failed**:
- **Vision-R1**: Vision-R1-Zero (RL only) barely improves over base (+1.1% with curriculum, vs +5.8% with cold start + curriculum)
- **ReVPT**: R1-Zero approach led to "progressive decline in tool usage" — model learns to avoid tools entirely
- **VTool-R1**: Untrained models prompted for tool use drop catastrophically (51.8 -> 24.6 accuracy)
- **CogCoM**: Uses 70K SFT samples; never attempts RL at all

The consensus is overwhelming: **a model needs a tool-calling prior before RL can refine it.** Our forced prefix curriculum is a creative alternative to SFT, but Vision-R1's ablation shows that curriculum/scaffolding alone (without a prior) gives only marginal gains. The forced prefix may buy enough signal for GRPO to work — this is untested territory none of these papers explore — but if it doesn't, SFT warm-up is the proven fallback.

### 2. Outcome-only reward beats process reward
- **VTool-R1** (most explicit): Process rewards for tool calls cause either tool avoidance (penalizing bad calls) or reward hacking (rewarding good calls). They explicitly tried and rejected both.
- **ReVPT**: Binary +1/-1 (format + correctness). No partial credit.
- **Vision-R1**: Binary (format + correctness). No process reward model.

Our `reward_bbox.py` already follows this — reward is based on final answer correctness + IoU. The partial-credit IoU component is more granular than binary but still outcome-based (it measures the quality of the final tool call result, not whether the model attempted a tool call). This is fine and arguably better — more signal for GRPO's advantage estimation.

**Do NOT add**: rewards for "valid JSON", "attempted tool call", or other process-level signals. All papers agree this backfires.

### 3. ~50-200 RL steps suffice for convergence
- **VTool-R1**: ~50 steps (one epoch)
- **Vision-R1**: 200 steps (100 per PTST stage)
- **ReVPT**: 150-200 steps

Our curriculum phases of 20 steps each are likely too short. If Apertus needs 50+ steps just to refine an already-present behavior, and it's starting from a harder position (forced prefix, not SFT), we should expect needing at least 50-100 steps per phase, or 200+ total steps.

### 4. Small data is enough
- **CogCoM**: 70K SFT samples, but shows good results at 2K training steps
- **ReVPT**: 1.5K cold-start SFT examples + 20K RL questions
- **Vision-R1**: 200K cold-start + only 10K RL examples
- **VTool-R1**: Standard training set, converges at ~50 steps

For Apertus SFT warm-up (if needed): even 50-100 examples should be enough. CogCoM's data pipeline and ReVPT's GPT-4.1 trajectory generation are both viable methods to create these.

### 5. Tool use frequency is non-monotonic — don't panic
- **VTool-R1**: Models overuse tools early, then become selective
- **ReVPT**: Object detection and edge detection usage decreases after RL
- Both show: higher accuracy when tools ARE used, meaning the model learns WHEN to use them

If Apertus starts making tool calls and then the frequency drops, this is expected and healthy — as long as accuracy improves.

### 6. All four papers use veRL or similar GRPO infrastructure
VTool-R1, Vision-R1, and ReVPT all use veRL. CogCoM is SFT-only but validates the multi-turn architecture. Our infrastructure choice is well-validated.

---

## Contradictions / Open Questions

### CogCoM's internalized tools vs everyone else's external tools
CogCoM generates both the tool call AND the result internally (no external execution). Everyone else uses external tool execution. CogCoM avoids the cold-start problem (model learns the full pattern from SFT) but sacrifices accuracy (hallucinated results). External tools are more accurate but create the bootstrapping problem. **No paper explores the middle ground**: could we start with internalized/hallucinated tool results during SFT, then switch to external execution during RL?

### Binary reward vs our IoU-based reward
Three papers use strict binary rewards and succeed. Our reward function gives partial credit via IoU. Neither approach has been tested on a zero-prior model. Binary may be too sparse for Apertus (if it rarely produces anything correct, there's no signal). Our IoU-based reward provides more gradient signal but risks the process-reward problems VTool-R1 warns about. **Untested question**: is IoU-based reward "outcome enough" to avoid reward hacking, or is it too close to a process reward?

### How much cold-start data is really needed for zero-prior models?
All papers tested on Qwen2.5-VL which already has instruction-following and some tool-calling priors. Apertus has zero. ReVPT used 1.5K examples for Qwen — Apertus might need more, or the examples might need to be more explicit. **No paper addresses the truly zero-prior case.**

### VTool-R1's single-turn vs ReVPT's multi-turn
VTool-R1 limits to one tool call and gets strong results. ReVPT allows up to 5 turns. Our setup is effectively 2 turns (call tool, get result, answer). The optimal number of turns for a zero-prior model is unclear — more turns = more exploration space = harder to learn.

---

## Ideas for Our Project

### Immediate (test with current codebase)
1. **Extend curriculum phases to 50+ steps each** (total 200+ steps). All papers show convergence at 50-200 steps. Our 20-step phases are almost certainly too short. Change in `tool_agent_loop.py:_get_apertus_forced_prefix_ids()` and `emu3_bbox_grpo.yaml:total_training_steps`.

2. **Run the reward decode fix test first**. The `skip_special_tokens=False` fix hasn't been validated yet. Before changing anything else, submit a job and check `reward_debug_<JOBID>.jsonl` to confirm the reward function now sees tool tokens.

### If forced prefix alone doesn't work (SFT warm-up path)
3. **Generate 50-100 SFT examples using GPT-4**. Following CogCoM's pipeline / ReVPT's approach:
   - Take training images + questions
   - Use GPT-4 to generate the correct `<|tools_prefix|>[{"image_bbox_tool": {"bbox_2d": [x1,y1,x2,y2]}}]<|tools_suffix|>` call
   - Execute the tool, get the result
   - Generate the final `<boxed>answer</boxed>`
   - Filter for correct answers only
   - SFT Apertus for 1-2 epochs on these before GRPO

4. **Consider Vision-R1's PTST for after SFT**: Start RL with shorter max_tokens (force concise tool calls) and more rollouts, then gradually increase max_tokens and decrease rollouts. Prevents the overthinking problem Vision-R1 identified post-SFT.

### Longer-term ideas
5. **Adaptive curriculum based on reward signal** (CLAUDE.md alternative #4): Instead of fixed step-count decay, monitor `critic/score/mean` and only advance to the next curriculum phase when rewards plateau. More robust than fixed schedules.

6. **Feed cropped image back as visual input** instead of text description. VTool-R1 and ReVPT both return tool results as images, not text. Apertus uses IBQ tokens for images, so we could encode the cropped region and inject it as visual tokens. This is architecturally harder but more faithful to how these other systems work.

7. **Data filtering**: Following ReVPT, filter training data to keep questions Apertus gets wrong without tools. This ensures GRPO rollouts produce diverse outcomes (some correct, some not), which is critical for advantage estimation.
