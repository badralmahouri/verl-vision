#!/usr/bin/env python3
"""Extract metrics from TensorBoard and create plots."""

import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

# Find the most recent events file
tb_dir = "tensorboard_log/multimodal_tool_example/image_line_tool_single_gpu_example"
events_files = [f for f in os.listdir(tb_dir) if f.startswith('events.out.tfevents')]
events_path = os.path.join(tb_dir, sorted(events_files)[-1])

print(f"Reading from: {events_path}")

# Load events
ea = event_accumulator.EventAccumulator(events_path)
ea.Reload()

# Get all scalar tags
tags = ea.Tags()['scalars']
print(f"\nAvailable metrics ({len(tags)} total):")
for tag in sorted(tags)[:10]:
    print(f"  - {tag}")
print("  ...")

# Extract key metrics
metrics = {}
for tag in tags:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    metrics[tag] = {'steps': steps, 'values': values}

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('VERL Training Progress - Image Line Tool', fontsize=16)

# Plot 1: Validation Accuracy
if 'val-core/line/acc/mean@1' in metrics:
    ax = axes[0, 0]
    data = metrics['val-core/line/acc/mean@1']
    ax.plot(data['steps'], data['values'], 'b-', linewidth=2, marker='o')
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.grid(True, alpha=0.3)
    if data['values']:
        ax.axhline(y=data['values'][0], color='r', linestyle='--', alpha=0.5, label='Baseline')
        ax.legend()

# Plot 2: Critic Scores
ax = axes[0, 1]
if 'critic/score/mean' in metrics:
    data = metrics['critic/score/mean']
    ax.plot(data['steps'], data['values'], 'g-', linewidth=2)
    ax.fill_between(data['steps'], 
                     metrics.get('critic/score/min', {}).get('values', []),
                     metrics.get('critic/score/max', {}).get('values', []),
                     alpha=0.2)
ax.set_xlabel('Step')
ax.set_ylabel('Score')
ax.set_title('Critic Scores (mean ± range)')
ax.grid(True, alpha=0.3)

# Plot 3: Actor Loss
ax = axes[1, 0]
for tag in ['actor/pg_loss', 'actor/kl_loss']:
    if tag in metrics:
        data = metrics[tag]
        ax.plot(data['steps'], data['values'], linewidth=2, label=tag.split('/')[-1])
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Actor Losses')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Throughput / Performance
ax = axes[1, 1]
if 'perf/throughput' in metrics:
    data = metrics['perf/throughput']
    ax.plot(data['steps'], data['values'], 'purple', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens/sec')
    ax.set_title('Training Throughput')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: training_progress.png")

# Print summary
current_step = max([max(m['steps']) for m in metrics.values() if m['steps']])
print(f"\n=== Training Summary ===")
print(f"Current step: {current_step}")
if 'val-core/line/acc/mean@1' in metrics:
    val_data = metrics['val-core/line/acc/mean@1']
    if val_data['values']:
        print(f"Validation accuracy: {val_data['values'][-1]:.2%} (baseline: {val_data['values'][0]:.2%})")
if 'critic/score/mean' in metrics:
    score_data = metrics['critic/score/mean']
    if score_data['values']:
        print(f"Latest critic score: {score_data['values'][-1]:.4f}")

plt.show()
