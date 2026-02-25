#!/usr/bin/env python3
"""
Test script to verify reward_std_patch works correctly.
Tests the patched compute_data_metrics function with mock data.
"""

import sys
import os

# Add path for our patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the patch BEFORE verl (this is the key)
import reward_std_patch

import torch
from verl import DataProto
from tensordict import TensorDict


def create_mock_batch(batch_size=32, response_length=100):
    """Create a mock batch similar to what verl's trainer produces."""
    
    # Create mock rewards with known statistics for verification
    token_level_scores = torch.randn(batch_size, response_length)
    token_level_rewards = torch.randn(batch_size, response_length)
    
    # Advantages and returns
    advantages = torch.randn(batch_size, response_length)
    returns = torch.randn(batch_size, response_length)
    
    # Attention mask (1 for valid tokens, 0 for padding)
    prompt_length = 50
    total_length = prompt_length + response_length
    attention_mask = torch.ones(batch_size, total_length)
    # Add some padding at the end for some samples
    for i in range(batch_size // 4):
        attention_mask[i, -20:] = 0
    
    # Response mask
    response_mask = attention_mask[:, -response_length:]
    
    # Responses tensor
    responses = torch.randint(0, 1000, (batch_size, response_length))
    
    batch_dict = {
        "token_level_scores": token_level_scores,
        "token_level_rewards": token_level_rewards,
        "advantages": advantages,
        "returns": returns,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "responses": responses,
    }
    
    td = TensorDict(batch_dict, batch_size=[batch_size])
    return DataProto(batch=td)


def test_patch():
    """Test that compute_data_metrics now includes std."""
    from verl.trainer.ppo.metric_utils import compute_data_metrics
    
    print("=" * 60)
    print("Testing reward_std_patch")
    # Create mock batch
    batch = create_mock_batch(batch_size=64, response_length=128)
    # Compute metrics
    metrics = compute_data_metrics(batch, use_critic=False)
    
    print("\nMetrics computed:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.6f}")
    
    # Verify std fields exist
    has_reward_std = "critic/rewards/std" in metrics
    has_score_std = "critic/score/std" in metrics
    
    print("\n" + "=" * 60)
    if has_reward_std and has_score_std:
        print("✓ SUCCESS: Both critic/rewards/std and critic/score/std are logged!")
        print(f"  - critic/rewards/std = {metrics['critic/rewards/std']:.6f}")
        print(f"  - critic/score/std = {metrics['critic/score/std']:.6f}")
    else:
        print("✗ FAILED: Missing std metrics")
        if not has_reward_std:
            print("  - Missing: critic/rewards/std")
        if not has_score_std:
            print("  - Missing: critic/score/std")
    print("=" * 60)
    
    return has_reward_std and has_score_std


if __name__ == "__main__":
    success = test_patch()
    sys.exit(0 if success else 1)
