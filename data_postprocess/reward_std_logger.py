#!/usr/bin/env python3
"""
Lightweight reward std logger for verl.

Usage without modifying verl:
    1. Wrap your reward function to track std
    2. Or post-process tensorboard logs

This module provides both approaches.
"""

import torch
from typing import Callable, Optional
from functools import wraps


class RewardStdTracker:
    """
    Lightweight wrapper to track reward statistics including std.
    
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards_buffer = []
    
    def update(self, rewards: torch.Tensor) -> dict:
        """Update with batch rewards and return current stats."""
        # Flatten and store
        flat_rewards = rewards.detach().flatten().cpu().tolist()
        self.rewards_buffer.extend(flat_rewards)
        
        # Keep only recent samples
        if len(self.rewards_buffer) > self.window_size:
            self.rewards_buffer = self.rewards_buffer[-self.window_size:]
        
        return self.get_metrics()
    
    def get_metrics(self) -> dict:
        """Compute current statistics including std."""
        if not self.rewards_buffer:
            return {}
        
        t = torch.tensor(self.rewards_buffer)
        return {
            "critic/rewards/std": t.std().item(),
            "critic/rewards/var": t.var().item(),
        }
    
    def reset(self):
        self.rewards_buffer = []


def wrap_reward_fn(reward_fn: Callable, tracker: RewardStdTracker) -> Callable:
    """
    Wrap a reward function to automatically track statistics.
    
    """
    @wraps(reward_fn)
    def wrapped(*args, **kwargs):
        result = reward_fn(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            tracker.update(result)
        return result
    return wrapped

#std logger for tensorboard logs
def compute_reward_std_from_tb(logdir: str, output_logdir: Optional[str] = None):
    """
    Read tensorboard logs and compute reward std from min/max/mean.
    
    Note: This is an approximation. For accurate std, use the tracker approach.
    
    Args:
        logdir: Path to tensorboard log directory
        output_logdir: Where to write new logs with std (default: logdir + '_with_std')
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("Install tensorboard: pip install tensorboard")
        return
    
    import os
    import glob
    
    output_logdir = output_logdir or f"{logdir}_with_std"
    
    # Find event files
    event_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    
    for event_file in event_files:
        print(f"Processing: {event_file}")
        
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Get reward metrics
        scalars = ea.Tags()['scalars']
        reward_metrics = [s for s in scalars if 'reward' in s.lower()]
        
        if 'critic/rewards/mean' in scalars:
            mean_events = ea.Scalars('critic/rewards/mean')
            
            # Create output writer
            rel_path = os.path.relpath(os.path.dirname(event_file), logdir)
            out_dir = os.path.join(output_logdir, rel_path)
            os.makedirs(out_dir, exist_ok=True)
            
            writer = SummaryWriter(out_dir)
            
            # Estimate std from range (rough approximation: std ≈ range/4 for normal dist)
            if 'critic/rewards/max' in scalars and 'critic/rewards/min' in scalars:
                max_events = ea.Scalars('critic/rewards/max')
                min_events = ea.Scalars('critic/rewards/min')
                
                for mean_e, max_e, min_e in zip(mean_events, max_events, min_events):
                    # Rough std estimate from range
                    range_val = max_e.value - min_e.value
                    std_estimate = range_val / 4.0  # ~95% within 2 std
                    
                    writer.add_scalar('critic/rewards/std_estimate', std_estimate, mean_e.step)
            
            writer.close()
            print(f"  Written to: {out_dir}")



# Extended metric computation for verl trainer
def extended_compute_data_metrics(batch, use_critic: bool = True) -> dict:
    """
    Extended version of verl's compute_data_metrics that includes std.
    
    This shows what fields to add. You can monkey-patch or create a custom
    trainer that calls this.
    """
    # Import original function
    from verl.trainer.ppo.metric_utils import compute_data_metrics
    
    # Get original metrics
    metrics = compute_data_metrics(batch, use_critic)
    
    # Add std for rewards
    response_length = batch.batch["responses"].shape[-1]
    response_mask = batch.batch["attention_mask"][:, -response_length:]
    response_length_per_sample = response_mask.sum(-1).float()
    non_aborted_mask = (response_length_per_sample > 0).bool()
    
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    non_aborted_rewards = sequence_reward[non_aborted_mask]
    
    if non_aborted_rewards.numel() > 1:
        metrics["critic/rewards/std"] = torch.std(non_aborted_rewards).detach().item()
        metrics["critic/rewards/var"] = torch.var(non_aborted_rewards).detach().item()
    
    # Also add score std
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    non_aborted_scores = sequence_score[non_aborted_mask]
    
    if non_aborted_scores.numel() > 1:
        metrics["critic/score/std"] = torch.std(non_aborted_scores).detach().item()
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute reward std from tensorboard logs")
    parser.add_argument("logdir", help="Tensorboard log directory")
    parser.add_argument("--output", "-o", help="Output directory (default: <logdir>_with_std)")
    args = parser.parse_args()
    
    compute_reward_std_from_tb(args.logdir, args.output)
