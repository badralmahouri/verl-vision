#!/usr/bin/env python3
"""
Monkey-patch verl to log reward std without modifying the library.

"""

import torch


def patch_verl_metrics():
    """Patch verl's compute_data_metrics to include std."""
    try:
        from verl.trainer.ppo import metric_utils
    except ImportError:
        print("verl not installed, skipping patch")
        return False
    
    _original_compute_data_metrics = metric_utils.compute_data_metrics
    
    def patched_compute_data_metrics(batch, use_critic: bool = True) -> dict:
        metrics = _original_compute_data_metrics(batch, use_critic)
        
        # Add std for rewards and scores
        response_length = batch.batch["responses"].shape[-1]
        response_mask = batch.batch["attention_mask"][:, -response_length:]
        response_len_per_sample = response_mask.sum(-1).float()
        non_aborted_mask = (response_len_per_sample > 0).bool()
        
        # Reward std
        seq_reward = batch.batch["token_level_rewards"].sum(-1)
        non_aborted_rewards = seq_reward[non_aborted_mask]
        if non_aborted_rewards.numel() > 1:
            metrics["critic/rewards/std"] = torch.std(non_aborted_rewards).detach().item()
        
        # Score std
        seq_score = batch.batch["token_level_scores"].sum(-1)
        non_aborted_scores = seq_score[non_aborted_mask]
        if non_aborted_scores.numel() > 1:
            metrics["critic/score/std"] = torch.std(non_aborted_scores).detach().item()
        
        return metrics
    
    metric_utils.compute_data_metrics = patched_compute_data_metrics
    print("Patched verl to log reward/score std")
    return True


# Auto-patch on import
patch_verl_metrics()


if __name__ == "__main__":
    # CLI mode: post-process existing tensorboard logs
    import sys
    if len(sys.argv) > 1:
        from reward_std_logger import compute_reward_std_from_tb
        compute_reward_std_from_tb(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print("Usage: python reward_std_patch.py <tensorboard_logdir> [output_dir]")
        print("Or:    import reward_std_patch  # at top of training script")
