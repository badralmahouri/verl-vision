# Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models
**Wenxuan Huang et al. (East China Normal University, CUHK, Xiaohongshu Inc.), ICLR 2026**

## Key Idea
RL alone (DeepSeek-R1-Zero style) fails to elicit complex reasoning in multimodal LLMs due to data scarcity and the absence of a reasoning prior. Vision-R1 solves this with a two-phase approach: (1) cold-start SFT on 200K multimodal chain-of-thought data distilled from DeepSeek-R1 via "Modality Bridging", then (2) GRPO with Progressive Thinking Suppression Training (PTST) that starts with short sequence limits and gradually relaxes them, preventing overthinking while building correct reasoning habits.

## Method

### Phase 1: Cold-Start Data Construction (Vision-R1-cold, 200K samples)
- **Problem**: DeepSeek-R1 is text-only, cannot process images directly.
- **Modality Bridging**: Feed image+question to an MLLM (Qwen2.5-VL-72B) to produce a "Pseudo-CoT" (caption + step-level reasoning). Then feed image+question+Pseudo-CoT back into the MLLM to get a richer, reasoning-aware description. This textual description is sent to DeepSeek-R1, which generates high-quality CoT with human-like cognitive processes (questioning, reflection, self-correction).
- **Data filtering**: Keep only samples where DeepSeek-R1's final answer matches ground truth. Rule-based filtering for logical consistency and semantic coherence.
- **SFT**: Train the base MLLM (Qwen2.5-VL) for 2 epochs on Vision-R1-cold to get Vision-R1-CI (Cold-start Initialized).

### Phase 2: RL Training with PTST
- **Optimizer**: GRPO (Group Relative Policy Optimization). Hyperparams: epsilon=0.2, beta=1e-2.
- **Reward**: Hard Formatting Result Reward Function (HFRRF) -- binary reward, score=1 only when BOTH format (`<think>...</think><answer>...</answer>`) and correctness are satisfied, else 0. No partial credit.
- **The Overthinking Problem**: After cold-start SFT, the model generates excessively long CoT. Correct reasoning concentrates in shorter chains, but the model defaults to long, mostly-wrong reasoning. Directly training with 16K context makes this worse -- the model produces longer outputs but doesn't improve accuracy.
- **PTST (Progressive Thinking Suppression Training)**:
  - Stage 1: Max 4K tokens, 16 samples per group, 100 steps. Forces model to compress reasoning, learn correct patterns in short form.
  - Stage 2: Max 8K tokens, 8 samples per group, 100 steps. Relaxes constraint so model can tackle harder problems with longer reasoning.
  - (Optional Stage 3: 16K tokens, 4 samples -- did not improve performance, not used in final model.)
  - Key insight: keep sampling x length constant across stages (16x4K = 8x8K = 4x16K = 64K total tokens per question).
- **Training data**: Only 10K multimodal math problems for RL (20K for 32B/72B).
- **Framework**: verl (HybridFlow) for GRPO training.

### What did NOT work
- **Vision-R1-Zero** (RL only, no cold start): Fails to produce complex CoT. Average output length only 1285 tokens. Avg accuracy 50.7%.
- **Vision-R1-Long** (cold start + RL with 16K directly): Overthinking problem. 3107 avg tokens but only 47.7% accuracy.
- **Zero+SFT+PTST** (SFT without CoT annotations then PTST): Catastrophic -- 39.8% accuracy. SFT on non-CoT data actively hurts.
- **PTST without cold start** (Zero+PTST): Only marginal gain over RL-only (51.8% vs 50.7%).

## Results

### Main numbers (Vision-R1-7B, base = Qwen2.5-VL-7B)
| Benchmark | Base | Vision-R1-7B | Delta |
|-----------|------|-------------|-------|
| MathVista (ALL) | 68.1 | 73.5 | +5.4 |
| MathVerse | 46.7 | 52.4 | +5.7 |
| MM-Math | 34.1 | 40.2 | +6.1 |
| Average | 49.6 | 55.4 | +5.8 |

- Vision-R1-7B achieves 73.5% on MathVista, only 0.4% below OpenAI O1.
- Vision-R1-32B: 76.4% MathVista, Vision-R1-72B: 78.2% MathVista.
- Cold-start dataset quality confirmed: Vision-R1-cold has 250x more "Wait" tokens, 75K+ "Hmm" tokens vs near-zero in prior datasets -- genuine self-reflection patterns.

### Ablation takeaways
- Cold start is essential. PTST alone without cold start barely helps (+1.1% avg).
- PTST is essential. Cold start without PTST leads to overthinking and performance regression under RL.
- Two stages suffice. Adding a third stage or more samples in Stage 2 gives no meaningful gain.
- Binary reward (format + correctness) works. No process reward model needed.

## Relevant to Our Project

### 1. The "Reward Desert" = Their "Vision-R1-Zero" failure (DIRECT PARALLEL)
They found that RL alone on MLLMs fails when the model has no prior for the target behavior. Their Vision-R1-Zero struggled to produce complex CoT from scratch, just as Apertus fails to produce tool calls from scratch. Their solution: **cold-start SFT before RL**. This is exactly the "SFT warm-up" idea listed as alternative #1 in CLAUDE.md. Their data shows this is not optional -- PTST alone without cold start gains only +1.1%, but cold start + PTST gains +5.8%.

**Concrete recommendation**: Create 50-100 SFT examples of Apertus correctly calling `image_bbox_tool` with proper `<|tools_prefix|>...<|tools_suffix|>` format. SFT Apertus on these before starting GRPO. This paper strongly suggests the forced prefix curriculum alone will not be enough without this prior.

### 2. PTST maps directly to our Forced Prefix Curriculum
Their PTST progressively relaxes sequence length constraints. Our forced prefix curriculum progressively reduces the amount of forced structure. The philosophy is identical: scaffold heavily early, remove scaffolding as the model learns. However, there is a critical difference: **they compress length, we inject tokens**. Their approach works because the model already knows HOW to reason (from cold start) -- PTST just controls HOW MUCH. Our forced prefix approach assumes the model will learn WHAT to generate, which is harder. This reinforces the need for SFT warm-up first, then forced prefix decay.

### 3. Binary reward function works for GRPO
They use a strict binary reward: 1 if format AND answer correct, 0 otherwise. No partial credit, no IoU-based scoring, no process rewards. This worked well even with only 10K training examples. Our reward function in `reward_bbox.py` already does something similar (format match + answer correctness). The paper validates this design -- we do NOT need a more complex reward signal. However, the "partial JSON reward" idea (alternative #5 in CLAUDE.md) is not supported by this paper -- they found binary rewards sufficient when the model has a cold-start prior.

### 4. Small RL datasets are sufficient
They used only 10K math problems for RL training, 100 steps per PTST stage, 200 total steps. This suggests our dataset size is not a bottleneck. The quality of the cold-start prior matters much more than the RL dataset size.

### 5. The Overthinking Problem is a warning for us
After cold-start SFT, their model over-generated reasoning tokens. If we SFT Apertus on tool-calling examples, we may see a similar issue where the model produces tool-call-like tokens excessively. PTST-like length control during RL could help: start with short max generation lengths (force concise tool calls), then gradually increase.

### 6. Sampling x Length budget tradeoff
They keep total token budget per question constant across PTST stages (16 samples x 4K = 8 samples x 8K). More samples at shorter lengths in early stages gives better exploration when the model is still learning correct patterns. This applies to our GRPO setup: we could use more rollouts per prompt with shorter max_tokens in early training, then fewer rollouts with longer max_tokens once tool-calling behavior emerges.

### 7. verl is used as the training framework
They explicitly use verl (HybridFlow/EasyR1) for GRPO training. This validates our framework choice and means their training recipes are directly reproducible in our codebase.

## Limitations / Open Questions

1. **Only tested on Qwen2.5-VL and Llama-3.2-V**: Both are strong base models with good instruction-following priors. Apertus has a much weaker prior (image-text SFT only). The cold-start approach may need more data or more epochs for a model with zero tool-calling capability.

2. **Math reasoning only**: All experiments focus on math reasoning where the base model already has latent math knowledge. Tool calling is a qualitatively different behavior -- the model must learn a new output format and a new interaction pattern (call-then-respond), not just a new reasoning style. Cold start may need to be more heavily supervised.

3. **No multi-turn evaluation**: Vision-R1 generates a single response. Our setup is multi-turn: model generates tool call, receives tool output, generates final answer. The interaction between cold-start SFT and multi-turn RL is unexplored.

4. **Binary reward may not bootstrap from zero**: Their binary reward works because cold-start gives the model a >0 success rate from the start. If SFT quality is poor and the model rarely produces correct tool calls + correct answers, the reward signal will still be too sparse. May need a staged reward: first reward just for valid tool calls (format), then for correct bbox (IoU), then for correct final answer.

5. **No analysis of what tokens change under RL**: They track output length and accuracy but not which specific tokens or patterns the model learns/unlearns. For our tool-calling case, understanding whether the model learns the tool-call structure vs the bbox coordinates vs the final answer would be valuable.

6. **Overthinking suppression is response-length-based only**: They control max generation length. A more targeted approach might suppress specific patterns (e.g., repetitive reasoning loops). For our case, suppressing degenerate tool-call repetition (the `<|tools_prefix|><|tools_prefix|>...` problem from logit bias) might need token-level controls beyond length limits.
