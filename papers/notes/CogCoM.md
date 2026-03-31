# CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning
**Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, Jie Tang (Tsinghua University / Zhipu AI), 2024, ICLR 2025**

## Key Idea
VLMs trained with standard instruction tuning skip intermediate visual reasoning and jump to conclusions, causing failures on tasks requiring fine-grained detail. CogCoM introduces Chain of Manipulations (CoM) -- a mechanism where the model generates explicit step-by-step reasoning that includes *manipulations* on the image (grounding, crop-and-zoom, OCR, counting, calculating, drawing lines), with the model itself producing both the manipulation calls and their results, all within a single end-to-end model without relying on external tools at inference time.

## Method

### Manipulations (the "tool set")
Six atomic manipulations, defined during pilot experiments with GPT-4 on 170K TextVQA questions:
1. **OCR(target) -> text** -- recognize text in a region
2. **Grounding(target) -> bounding boxes** -- locate objects by description
3. **CropZoomIn(bbox, scale) -> new image** -- crop a region and enlarge it
4. **Counting(target) -> number** -- count objects
5. **Calculate(target) -> number** -- compute a formula
6. **Line(points) -> new image** -- draw auxiliary lines (for geometry)

The model can also create new manipulations on the fly during inference.

### Data Generation Pipeline (key innovation)
A cascading automated pipeline to produce 70K CoM training samples:
1. **Linguistic annotator** (GPT-4): Given a VQA question, generates step-by-step solving plans using manipulation calls, with variables as placeholders for results.
2. **Visual annotators** (GroundingDINO + PaddleOCR): Execute each manipulation to fill in actual results (bounding boxes, OCR text). This creates a tree structure since one manipulation may have multiple valid results.
3. **DFS traversal**: Search the tree for *positive paths* -- chains where the final result matches the gold answer. Only these paths become training data.
4. Success rate of GPT-4 generating a positive path: 35.55%.

Additionally, 7K high-quality CoM samples were manually annotated for graphical math problems (geometry, charts) by 10 human experts.

### Architecture
Built on CogVLM (17B parameters total: EVA2-CLIP-E 4B visual encoder + Vicuna-7B backbone + 6.5B visual expert module). Key architectural addition: **memory-based multi-turn multi-image support** -- KV cache accumulated across turns, with new cropped/zoomed images injected at each turn.

### Training Process
- **Stage 1-1**: Pre-training on 1.5B image-caption pairs (foundational visual understanding). 120K iterations, batch 8192.
- **Stage 1-2**: Grounded generation on 40M image-QA triples with bounding box annotations. 60K iterations, batch 1024.
- **Stage 2**: Alignment fine-tuning on 570K samples mixing CoM data (70K) with MultiInstruct, LLaVAR, ShareGPT4V. 14K iterations, batch 160. A "launching prompt" randomly prepended to questions tells the model it may use manipulations.

All training uses next-token prediction on 6.5B visual expert parameters.

### CoM Data Format
Standard structure: `(step1, step2, ...) where step_i = (manipulation_i, description_i)`. Converted to multi-turn VQA: `[(I0, Q, C0), (I1, Q_continue, C1), ..., (In, Q_continue, A)]` where each new image comes from a CropZoomIn manipulation.

## Results

### Detailed VQA (main strength)
- **GQA**: 71.7 (vs 65.2 baseline CogVLM, +6.5 points)
- **TextVQA**: 71.1 (vs 69.7 baseline, +1.4 points)
- **ST-VQA**: 70.0 (vs 61.0 baseline, +9.0 points)
- **TallyVQA-simple**: 84.0, **TallyVQA-complex**: 70.1

### Visual Grounding
Best among generalist models on 6/8 RefCOCO subsets. On par with specialist SOTAs.

### General / Hallucination
- MM-Vet: 46.1 (vs 45.5 baseline, +0.6)
- POPE adversarial: 87.8 (vs 87.2 baseline)

### Key Findings
- CoM reasoning learned effectively with only 70K training samples and ~2K steps (Figure 6).
- Ablation: removing CoM data drops TextVQA from 71.1 to 64.5, MMVet from 46.1 to 43.6.
- Time overhead is modest: ~262 tokens in ~9 seconds, vs baseline's ~141 tokens.
- The model learns to selectively use manipulations -- not every question triggers CoM.

## Relevant to Our Project

### 1. The "internalized tool" approach vs our external tool approach
CogCoM trains the model to *internalize* manipulations -- the model generates both the tool call AND the tool result. In our verl-vision setup, the model generates the tool call, an *external* tool executes it, and the result is injected back. CogCoM avoids the execution gap but sacrifices accuracy (model hallucinates bounding boxes); our approach gets exact tool results but faces the cold-start problem of teaching the model to call the tool at all.

**Takeaway**: CogCoM's approach would not work for Apertus because it requires the model to already understand the manipulation semantics (learned from 70K SFT samples). Our external-tool approach is correct for the zero-prior setting, but we need to solve the bootstrapping problem.

### 2. Data generation pipeline as an alternative to RL bootstrapping
CogCoM's biggest contribution is the automated pipeline: GPT-4 generates reasoning plans, visual tools fill in results, DFS finds valid paths. This is essentially **synthetic SFT data generation for tool use**. We could adapt this to generate Apertus-format tool-calling examples:
- Use GPT-4 to generate `<|tools_prefix|>[{"image_bbox_tool": {"bbox_2d": [x1,y1,x2,y2]}}]<|tools_suffix|>` calls for our training images.
- Execute the tool to get actual results.
- Filter for paths where the final answer matches ground truth.
- SFT Apertus on these ~50-100 examples before starting RL.

This directly addresses the **"reward desert" problem**: instead of hoping RL discovers tool-calling from scratch, we give the model a few working examples via SFT (idea #1 in CLAUDE.md's alternative approaches).

### 3. Launching prompts as curriculum
CogCoM randomly prepends "launching prompts" that explicitly tell the model it can use manipulations. This is analogous to our **forced prefix curriculum** but softer -- it's a natural language instruction rather than forced token injection. Their prompts like "Please solve the problem gradually via a chain of manipulations" are essentially what our `apertus_tool_chat_template.jinja2` tries to do.

**Key difference**: CogCoM's model has seen 70K examples of what manipulations look like, so a text prompt suffices. Apertus has seen zero, so we need the harder forced-prefix approach.

### 4. Multi-turn multi-image architecture
CogCoM's KV-cache accumulation across turns with new images injected at each turn is exactly our agent loop pattern: model generates -> tool executes and injects result -> model continues. Their architecture validates that this multi-turn approach works for visual reasoning with tool-like operations.

### 5. The 70K number as a data efficiency benchmark
CogCoM achieves strong CoM reasoning with only 70K SFT samples (and sees good results at just 2K training steps). This suggests that even a small number of high-quality tool-calling demonstrations could be enough for SFT warm-up of Apertus. We don't need thousands -- perhaps 50-200 well-crafted examples would give the model enough prior to make RL viable.

### 6. CropZoomIn is directly analogous to our image_bbox_tool
Their CropZoomIn(bbox, scale) manipulation is functionally identical to our `image_bbox_tool(bbox_2d)`. The model specifies a bounding box, the region is cropped and enlarged, and the model reasons about the cropped image. This validates our task design.

### 7. Positive-path filtering via DFS
Their DFS traversal to find chains that reach the gold answer is a form of **outcome-based filtering** -- only keep trajectories that work. This is conceptually similar to GRPO's approach of reinforcing trajectories with positive reward, but done at the data generation stage rather than during RL training.

## Limitations / Open Questions

1. **No RL training**: CogCoM is purely SFT. They mention in the future work section (Appendix D.1) that "using Reinforcement Learning to penalize the negative paths during training is another optimization strategy" -- they acknowledge RL could help but did not try it. Our project is exactly this: using RL (GRPO) to optimize tool-calling behavior.

2. **Hallucinated tool results**: Since the model generates both the tool call and result, it can hallucinate (produce wrong bounding boxes, wrong OCR text). External tool execution (our approach) avoids this.

3. **Data generation bottleneck**: The 35.55% success rate means ~65% of GPT-4 generations are wasted. The pipeline is expensive.

4. **Limited manipulation set**: Only 6 manipulations. No discussion of how to extend to arbitrary tools or APIs.

5. **No cold-start analysis**: CogCoM starts from CogVLM which already has strong grounding capabilities from Stage 1 pre-training. They never address the scenario where a model has zero prior for the manipulation type (our exact problem with Apertus).

6. **Backtracking mentioned but not implemented**: They discuss (Appendix B) that a backtracking mechanism during reasoning would help but has high time complexity. Left for future work.

7. **Scale**: 17B parameters, 3840 A100-days for Stage 1-1 alone. The training cost is substantial, though the CoM-specific Stage 2 is much cheaper (160 A100-days).
