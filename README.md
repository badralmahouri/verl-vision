# Multi-Modal Vision-Language Tool Learning with Reinforcement Learning

**CS-433 Machine Learning Project - EPFL**

This project extends the [verl](https://github.com/volcengine/verl) reinforcement learning framework to enable **multi-turn tool use training for Vision-Language Models (VLMs)**. We implement and train models to use visual manipulation tools (flip, blur, crop, line drawing, bounding box) through reinforcement learning with GRPO.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Data Generation](#data-generation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Reproducibility](#reproducibility)
8. [External Libraries and Datasets](#external-libraries-and-datasets)
9. [Acknowledgements](#acknowledgements)

---

## Project Overview

### Objective

Train Vision-Language Models (specifically Qwen2.5-VL) to learn **when and how to use image manipulation tools** through reinforcement learning. The model must:

1. Analyze an image and identify what transformation is needed
2. Use the appropriate tool with correct parameters
3. Provide the correct answer based on the transformed image

### Implemented Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `image_flip_tool` | Flip images horizontally/vertically | Reading mirrored text |
| `image_blur_tool` | Apply/remove blur (gaussian, box, motion) | Revealing blurred content |
| `image_crop_tool` | Crop regions from images | Focusing on specific areas |
| `image_line_tool` | Draw lines on images | Marking/highlighting regions |
| `image_bbox_tool` | Draw bounding boxes | Object localization |
| `image_rotate_tool`| Apply rotation with a given angle to the input image

### Key Contributions

- **5 new visual tools** with full integration into verl's multi-turn agent loop
- **Synthetic dataset generators** for each tool task
- **Custom reward functions** for verifiable tool-use evaluation
- **RefCOCO integration** for real-world object localization tasks
- **SLURM scripts** for HPC cluster training
- **Evaluation and comparison utilities**

---

## Repository Structure

```
verl-main/
├── data/                           # Dataset generation scripts
│   ├── generate_random_flip.py     # Flip task dataset generator
|   |── generate_flip_rotate.py     # Generates the rotate + flip dataset
│   ├── generate_random_blur.py     # Blur task dataset generator
│   ├── generate_random_bbox.py     # Bounding box task generator
│   ├── generate_random_line.py     # Line drawing task generator
│   ├── generate_random_zoom.py     # Zoom/crop task generator
│   ├── generate_real_world.py      # RefCOCO dataset preparation
│   └── generate_mixed_tools_dataset.py  # Multi-tool dataset
│
├── verl/tools/                     # Tool implementations
│   ├── image_flip_tool.py          # Flip tool (horizontal/vertical)
│   ├── image_blur_tool.py          # Blur/unblur tool
│   ├── image_crop_tool.py          # Crop tool
│   ├── image_line_tool.py          # Line drawing tool
│   └── image_rotate_tool.py        # Rotation tool
│
├── verl/utils/reward_score/        # Reward functions
│   ├── reward_flip.py              # Flip task rewards
│   ├── reward_blur.py              # Blur task rewards
│   ├── reward_crop.py              # Crop task rewards (IoU-based)
│   ├── reward_line.py              # Line task rewards
│   ├── reward_bbox.py              # Bounding box rewards
│   └── reward_refcoco.py           # RefCOCO evaluation rewards
│
├── examples/sglang_multiturn/config/  # Training configurations
│   ├── flip_grpo.yaml              # Flip task GRPO config
│   ├── blur_grpo.yaml              # Blur task GRPO config
│   ├── bbox_grpo.yaml              # Bbox task GRPO config
│   ├── line_grpo.yaml              # Line task GRPO config
│   ├── refcoco_crop_grpo.yaml      # RefCOCO crop training
│   ├── refcoco_bbox_grpo.yaml      # RefCOCO bbox training
│   └── multimodal_tool_config.yaml # Tool definitions
│
├── slurm/                          # SLURM job scripts
│   ├── run_image_flip_tool_example.slurm
│   ├── run_image_blur_tool_example.slurm
│   ├── run_image_bbox_tool_example.slurm
│   ├── run_image_line_tool_example.slurm
│   ├── run_refcoco_crop.slurm
│   ├── run_refcoco_bbox.slurm
│   ├── compare_eval.py             # Baseline vs trained comparison
│   └── plot.py                     # Plotting utilities
│
├── data_postprocess/               # Metrics and logging
│   ├── reward_std_logger.py        # Reward standard deviation logging
│   ├── verl_metrics.py             # Custom metrics extraction
│   └── test_std_patch.py           # Testing utilities
│
└── plot_tensorboard.py             # TensorBoard visualization
```

---

## Installation

### Prerequisites

- Python 3.10
- CUDA 12.4+
- Conda

### Setup

```bash
# Clone the repository
git clone https://github.com/Atemis8/verl-main.git
cd verl-main

# Create conda environment
conda create -n verl python=3.10
conda activate verl

# Install dependencies
# Note that you may need to install custom wheels for your cluster
pip install -e .
pip install -r requirements.txt

# Install additional dependencies for VLM support
pip install qwen-vl-utils pillow pyarrow
```

### Environment Variables

```bash
export HF_HOME=/path/to/huggingface/cache
export CUDA_VISIBLE_DEVICES=0  # or appropriate GPU IDs
```

---

## Data Generation

### Synthetic Datasets

Generate training data for individual tools:

```bash
cd data/

# Generate flip task dataset (mirrored text reading)
python generate_random_flip.py \
    --output_dir ./flip \
    --num_samples 5000 \
    --split_ratio 0.9

# Generate blur task dataset
python generate_random_blur.py \
    --output_dir ./blur \
    --num_samples 5000

# Generate bounding box task dataset
python generate_random_bbox.py \
    --output_dir ./bbox \
    --num_samples 5000

python generate_random_zoom.py \
    --output_dir ./zoom
    --num_samples 5000

# Generate line drawing task dataset
python generate_random_line.py \
    --output_dir ./line \
    --num_samples 5000
```

### RefCOCO Real-World Dataset

For real-world object localization tasks:

```bash
python generate_real_world.py \
    --task bbox \
    --output_dir ./refcoco_bbox \
    --num_samples 10000

python generate_real_world.py \
    --task crop \
    --output_dir ./refcoco_crop \
    --num_samples 10000
```

### Dataset Format

Generated datasets are in Parquet format with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | list[dict] | Chat-format messages with system prompt and user query |
| `answer` | string | Ground truth answer in `<boxed>answer</boxed>` format |
| `images` | list[string] | Base64-encoded images or file paths |

---

## Training
Different premade scripts for the different trainings can be found in the /slurm folder
### Single-Tool Training

Train on individual tool tasks using GRPO, they do require a HPC environment :

```bash
# Submit flip tool training job
sbatch slurm/run_image_flip_tool_example.slurm

# Submit RefCOCO crop training
sbatch slurm/run_refcoco_crop.slurm
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.total_training_steps` | 50-100 | Number of training iterations |
| `actor_rollout_ref.rollout.n` | 4 | Rollout samples per prompt |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | Actor learning rate |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | 5 | Max tool use turns |

---
### Pre-trained Checkpoints

Checkpoints can be saved during training with:
```yaml
trainer:
  save_freq: 10  # Save every 10 steps
  default_local_dir: ./checkpoints
```


## Evaluation
The slurm folder contains a script to run evaluation on the crop task, you can also convert a checkpoint using the convert checkpoint :
```
# Takes the BBOX checkpoint step as input and converts it to a usable format
sbatch slurm/convert_checkpoint.slurm 100

# Takes the model step as input, if none runs the baseline test with Qwen
sbatch slurm/run_crop_eval.slurm your_hf_model
```

### Metrics

- **Accuracy**: Correct answer in `<boxed></boxed>` tags
- **Tool Usage Rate**: Percentage of samples where tool was correctly invoked
- **IoU Score** (for bbox/crop): Intersection over Union with ground truth
- **Reward Standard Deviation**: Diversity of rewards across rollouts, this can be enabled using the post processing script found in data_postprocess (it adds additional metrics logs)

### TensorBoard Visualization

```bash
tensorboard --logdir ./outputs/tensorboard_log
```

---

## External Libraries and Datasets
These are the main libraries used, in pratice those each have many dependencies 
### Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| [verl](https://github.com/volcengine/verl) | base | RL training framework |
| [SGLang](https://github.com/sgl-project/sglang) | 0.5.6+ | Multi-turn rollout generation |
| [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3B | Vision-Language Model |
| [PyArrow](https://arrow.apache.org/docs/python/) | - | Dataset storage |
| [Pillow](https://pillow.readthedocs.io/) | - | Image manipulation |
| [Ray](https://docs.ray.io/) | 2.52+ | Distributed computing |

### Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| RefCOCO | [UNC Vision](https://github.com/lichengunc/refer) | Real-world object localization |
| COCO 2014 | [COCO Dataset](https://cocodataset.org/) | Images for RefCOCO |
| Synthetic | Generated | Tool-specific training tasks |

---

## Acknowledgements

This project builds upon:
- **[Jakhongir0103](https://github.com/Jakhongir0103/verl)**: The base changes to support multimodal processing (previous work from the Swiss AI team)
- **[verl](https://github.com/volcengine/verl)**: The base reinforcement learning framework by ByteDance Seed Team
- **[SGLang](https://github.com/sgl-project/sglang)**: For efficient multi-turn generation
- **[Qwen2.5-VL](https://github.com/QwenLM)**: The vision-language model architecture

