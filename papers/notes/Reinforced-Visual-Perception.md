# Reinforced Visual Perception with Tools (ReVPT)
**Zetong Zhou, Dongping Chen, Zixian Ma, et al. (ONE Lab HUST, UW, UMD, ZJU). arXiv:2509.01656, Sep 2025.**

## Key Idea
ReVPT trains multi-modal LLMs (Qwen2.5-VL 3B/7B) to use four visual tools (object detection, zoom-in, edge detection, depth estimation) via reinforcement learning (GRPO), rather than relying solely on supervised fine-tuning. The core insight is that RL enables the model to explore and discover its own tool-use strategies, outperforming SFT on perception-heavy benchmarks by significant margins (9%+ on CV-Bench).

## Method

### Two-Stage Training
1. **Cold-start SFT** (critical -- RL alone fails): GPT-4.1 synthesizes ~1.5k tool-use trajectories (reasoning + tool call + observation + answer). Model is SFT'd on these for 2 epochs (lr=1e-5, batch=64). Without cold start, the model's tool usage "progressively declines" because solving visual tasks doesn't inherently require tools -- the model takes the easier path.
2. **GRPO RL** (~200 steps to convergence): 20k questions filtered to be hard for Qwen2.5-VL-7B (it gets them wrong). The model explores tool use during multi-turn rollouts (up to 5 turns, 1024 tokens/turn). lr=2e-6, KL coef=1e-3, 8 generations per prompt, mini-batch=128.

### Visual Tools (4 tools)
- **Object Detection** (open-vocabulary, LLMDet): image + text query -> annotated image + bounding boxes
- **Zoom In**: image + bbox + magnification factor -> cropped/magnified image
- **Edge Detection** (Scharr algorithm): image -> edge map
- **Depth Estimation** (DepthAnything v2): image -> colored depth map

Tool outputs are **images** fed back as new visual inputs in subsequent turns (interleaved image-text reasoning traces).

### Reward Design
Binary reward, no learned reward model:
- **+1** if format is correct AND answer is correct
- **-1** otherwise

Format requires: `<think>...</think>` for reasoning, `<tool_call>...</tool_call>` for tool invocation, `<answer>...</answer>` for final response.

### Data Construction
- Source: SAT (spatial reasoning VQA) + TACO (tool-augmented CoT data)
- Filter: Keep only questions Qwen2.5-VL-7B gets wrong (ensures GRPO sampling produces both correct and incorrect rollouts)
- Cold-start subset: GPT-4.1 generates tool-use trajectories, filter to correct-answer-only
- RL subset: Questions only, no trajectories needed

### Tool Format
Standard JSON tool calls: `<tool_call>{"name": "object_detection", "arguments": {"image_id": 0, "objects": ["tie"]}}</tool_call>`

Tool results injected as: `<result>Detected 4 object(s)...</result>` with optional `<image>` for visual outputs.

### Agent Config
- max_turns: 5
- max_tokens_per_turn: 1024
- inference_batch_size: 8 (limited due to vLLM quality degradation at larger batches)

## Results

### Main Numbers (vs Qwen2.5-VL-Instruct baseline)
| Model | CV-Bench | BLINK | MMVP | MMStar | Avg (8 benchmarks) |
|-------|----------|-------|------|--------|---------------------|
| ReVPT-3B | 81.20 (+8.65) | 72.35 (+8.06) | 68.70 (+6.03) | 53.87 (+0.47) | 65.67 (+6.69) |
| ReVPT-7B | 84.23 (+9.82) | 73.64 (-6.77) | 72.00 (-2.00) | 61.07 (-0.73) | 69.42 (+5.73*) |

*7B delta is vs instruct, actual improvements vary by benchmark.

### Key Comparisons
- **ReVPT > SFT cold-start** across most benchmarks -- RL adds value on top of SFT
- **ReVPT > text-only GRPO** (no tools) -- visual tool feedback genuinely helps
- **ReVPT-3B and 7B > GPT-4.1** on BLINK-Depth and BLINK-Relation subsets
- Reward converges around step 150-200 for ReVPT; text-only GRPO converges lower

### Ablation Insights
- Removing object detection causes 5-12 point drops on relation/spatial tasks
- Cold-start data quality matters: including "hints" about when/why to use tools preserves general capabilities
- Including general VQA data (from TACO) during cold-start prevents catastrophic forgetting
- All 4 tools contribute; object detection and depth estimation are most frequently used

### Tool Usage Patterns After RL
- Object detection and edge detection usage **decreases** after RL (model learns when NOT to use tools)
- Tool usage is benchmark-dependent: high on perception benchmarks (CV-Bench ~62%, MMVP ~71%), low on general benchmarks (MMStar ~18%)
- Model achieves higher accuracy when it does choose to use tools, indicating learned selectivity

### Failure Modes
1. **Incorrect tool output**: Tool misclassifies objects, model trusts it blindly
2. **Incorrect tool interpretation**: Tool returns correct output but model misreads it
3. **Inappropriate tool usage**: Tool output interferes with correct reasoning
4. **Wrong tool selection**: Model picks irrelevant tool for the task

## Relevant to Our Project

### 1. Cold-start is essential -- validates our forced-prefix curriculum approach
ReVPT tried R1-Zero (RL from scratch, no SFT) and it **failed** -- "we observe a progressive decline in the agent's propensity to utilize tools." This is exactly our "reward desert" problem with Apertus. Their solution was SFT cold-start with 1.5k GPT-4.1 trajectories. Our forced-prefix curriculum is an alternative to SFT cold-start: instead of teaching the model tool-call patterns via SFT, we force the prefix tokens and let RL teach the rest. Both approaches solve the same bootstrap problem.

**Concrete takeaway**: If forced-prefix curriculum doesn't yield results after testing the reward decode fix, a small SFT warm-up (50-100 examples, as mentioned in CLAUDE.md alternative ideas) is strongly supported by this paper. ReVPT used only 1.5k cold-start examples and that was sufficient.

### 2. Binary reward (+1/-1) works -- our reward design is on the right track
ReVPT uses a simple binary reward: +1 for correct format AND correct answer, -1 otherwise. No partial credit for tool calls, no IoU-based scoring, no intermediate rewards. This is simpler than our reward_bbox.py which computes IoU and gives partial credit. Their success with binary rewards suggests our more nuanced reward might actually be fine (more signal = faster learning), but if debugging is hard, simplifying to binary could work too.

### 3. Data filtering strategy: use questions the model gets wrong
ReVPT filters training data to keep only questions the base model answers incorrectly. This ensures GRPO sampling produces diverse rollouts (some correct, some not), which is critical for advantage estimation. We should verify our training data has this property -- if Apertus already gets most bbox questions wrong (likely, given zero tool prior), this is naturally satisfied.

### 4. Tool outputs as images, not text
ReVPT tools return **images** (depth maps, annotated images, edge maps) that go back into the model as visual tokens. Our `image_bbox_tool` returns a cropped image region as text description. This is a design difference worth noting -- ReVPT's approach is richer but requires the model to process interleaved image-text, which Apertus handles via IBQ tokens.

### 5. 200 RL steps to convergence
ReVPT converges at ~200 steps with 20k training questions. Our forced-prefix curriculum spans 60 steps before removing scaffolding. If the reward signal works, we might need to run significantly longer (200+ steps) to see full convergence.

### 6. The "tool avoidance" problem
Even with Qwen2.5-VL (which has tool-calling capability), the model tends to stop using tools during RL because "solving visual tasks did not inherently require tool usage." This is a fundamental tension: RL optimizes for reward, and if the model can sometimes get rewards without tools, it will learn to skip them. Our setup forces tool use via the bbox task design (you can't answer without seeing the region), which is actually better aligned than ReVPT's setup where tools are optional.

### 7. Platform: they use veRL too
ReVPT uses veRL as their RL platform, same as us. This validates our infrastructure choice and means their training configs (Table 6) are directly comparable to ours.

### 8. Inference batch size degradation
They note "degradation in response quality when batch-inferencing with Qwen2.5-VL by vllm as batch size increased" and capped inference_batch_size at 8. We should watch for similar issues with SGLang + Apertus.

### 9. Format reward is important
Their reward requires correct format (`<think>`, `<tool_call>`, `<answer>` tags) in addition to correct answers. We check for `<boxed>answer</boxed>` format. This dual requirement (format + correctness) helps the model learn structured outputs.

### 10. General capability tradeoff
Training on perception-specific data degrades general capabilities. Their SFT-on-SAT baseline showed "substantial degradation across several capabilities." Including diverse data during cold-start mitigated this. Less relevant for us (we only care about bbox tool use), but worth knowing if we expand to more tasks.

## Limitations / Open Questions

1. **Only tested on Qwen2.5-VL** which already has strong tool-calling priors from instruct tuning. Would this work on a model with zero tool-calling prior (like our Apertus)? The cold-start SFT might not be enough for a truly naive model.

2. **1.5k cold-start examples require GPT-4.1** -- expensive and not always feasible. Our forced-prefix approach is cheaper (no data generation needed). But if forced-prefix fails, we'd need to generate SFT data.

3. **Binary reward may be too sparse** for models that rarely produce correct answers. ReVPT starts from a model that gets ~60% on many benchmarks. Apertus gets ~0% on tool-calling tasks initially. The reward signal may be too sparse for our case, supporting the idea of partial credit (IoU-based rewards, format-only rewards).

4. **Tool output quality limits the ceiling**: Their failure analysis shows tools can return wrong results, and the model can misinterpret correct results. The tool itself becomes a bottleneck.

5. **No exploration of curriculum learning for tool use**: They jump straight from cold-start SFT to full RL. A graduated curriculum (like our forced-prefix decay) is not explored. This is a gap their paper leaves open.

6. **Small tool suite (4 tools)**: They tried more tools but "found extremely low utilization rates" for smaller models. This suggests that training models to use many tools simultaneously is hard, especially for smaller models.

7. **200 steps with 20k data on 8xA800**: Their compute budget is substantial. They don't discuss sample efficiency or how quickly the model learns tool use patterns vs general reasoning improvements.

8. **The "Bitter Lesson" discussion is insightful but unresolved**: They acknowledge that cold-start SFT injects human biases about tool selection, and that ideally models should discover tool-use strategies autonomously. But they don't offer a solution beyond more compute.
