import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file):
    """Parse training log file and extract metrics."""
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Each line is a step entry
    metrics_data = []
    for line in lines:
        if 'step:' not in line:
            continue
        metrics_data.append(line.strip())
    
    metrics = {
        'step': [],
        'reward_mean': [],
        'reward_std': [],
        'val_reward_mean': [],
        'kl_loss': [],
        'kl_coef': [],
        'entropy': [],
        'pg_loss': [],
        'grad_norm': [],
        'lr': [],
        'response_length_mean': [],
        'advantages_mean': [],
        'advantages_max': [],
        'advantages_min': [],
    }
    
    for line in metrics_data:
        # Extract step number
        step_match = re.search(r'step:(\d+)', line)
        if not step_match:
            continue
        
        step_num = int(step_match.group(1))
        metrics['step'].append(step_num)
        
        # Extract each metric with proper regex
        # Handles both raw values (e.g., 0.5) and numpy-wrapped values (e.g., np.float64(0.5))
        def extract_metric(pattern, default=np.nan):
            m = re.search(pattern, line)
            if m:
                value_str = m.group(1)
                # Handle np.float64(...) or np.int32(...) wrapped values
                np_match = re.search(r'np\.\w+\(([\d\.\-e]+)\)', value_str)
                if np_match:
                    return float(np_match.group(1))
                return float(value_str)
            return default
        
        # Pattern that captures both raw values and np.float64(...) wrapped values
        # (?:np\.\w+\()? optionally matches "np.float64(" or "np.int32("
        # ([\d\.\-e]+) captures the numeric value
        # \)? optionally matches the closing ")"
        def make_pattern(base_pattern):
            """Create pattern that handles both raw and np-wrapped values."""
            return base_pattern + r'(?:np\.\w+\()?([\d\.\-e]+)\)?'
        
        metrics['reward_mean'].append(extract_metric(make_pattern(r'critic/rewards/mean:')))
        metrics['reward_std'].append(extract_metric(make_pattern(r'critic/rewards/std:')))
        # Generic pattern for validation reward - matches any data source (flip, bbox, etc.)
        metrics['val_reward_mean'].append(extract_metric(make_pattern(r'val-aux/\w+/reward/mean@1:')))
        metrics['kl_loss'].append(extract_metric(make_pattern(r'actor/kl_loss:')))
        metrics['kl_coef'].append(extract_metric(make_pattern(r'actor/kl_coef:')))
        metrics['entropy'].append(extract_metric(make_pattern(r'actor/entropy:')))
        metrics['pg_loss'].append(extract_metric(make_pattern(r'actor/pg_loss:')))
        metrics['grad_norm'].append(extract_metric(make_pattern(r'actor/grad_norm:')))
        metrics['lr'].append(extract_metric(make_pattern(r'actor/lr:')))
        metrics['response_length_mean'].append(extract_metric(make_pattern(r'response_length/mean:')))
        metrics['advantages_mean'].append(extract_metric(make_pattern(r'critic/advantages/mean:')))
        metrics['advantages_max'].append(extract_metric(make_pattern(r'critic/advantages/max:')))
        metrics['advantages_min'].append(extract_metric(make_pattern(r'critic/advantages/min:')))

    return metrics

def plot_training_metrics(metrics, save_dir='plots'):
    """Create comprehensive training plots."""
    
    Path(save_dir).mkdir(exist_ok=True)
    steps = np.array(metrics['step'])
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Rewards Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    reward_mean = np.array(metrics['reward_mean'])
    reward_std = np.array(metrics['reward_std'])
    
    ax.plot(steps, reward_mean, 'o-', label='Train Reward Mean', linewidth=2, markersize=6, color='blue')
    
    # Add std deviation shading if available
    valid_std_mask = ~np.isnan(reward_std)
    if np.any(valid_std_mask):
        ax.fill_between(steps[valid_std_mask], 
                        (reward_mean - reward_std)[valid_std_mask], 
                        (reward_mean + reward_std)[valid_std_mask], 
                        alpha=0.2, color='blue', label='Train Reward ±1 Std')
    
    # Add validation points
    val_steps = [s for s, v in zip(steps, metrics['val_reward_mean']) if not np.isnan(v)]
    val_rewards = [v for v in metrics['val_reward_mean'] if not np.isnan(v)]
    if val_rewards:
        ax.plot(val_steps, val_rewards, 's-', label='Val Reward Mean', 
                linewidth=2, markersize=8, color='red')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Reward / Score', fontsize=12)
    ax.set_title('Training and Validation Rewards', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rewards.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/rewards.pdf")
    
    # 2. KL Divergence and Entropy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(steps, metrics['kl_loss'], 'o-', label='KL Loss', linewidth=2, markersize=6)
    ax1.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Target Min (0.01)')
    ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Target Max (0.1)')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('KL Loss', fontsize=12)
    ax1.set_title('KL Divergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(steps, metrics['entropy'], 'o-', color='green', linewidth=2, markersize=6)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Entropy', fontsize=12)
    ax2.set_title('Policy Entropy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/kl_entropy.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/kl_entropy.pdf")
    
    # 3. Training Dynamics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(steps, metrics['pg_loss'], 'o-', color='purple', linewidth=2, markersize=6)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Policy Gradient Loss', fontsize=12)
    ax1.set_title('Policy Gradient Loss (Should decrease)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, metrics['grad_norm'], 'o-', color='red', linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Clip threshold (1.0)')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Gradient Norm', fontsize=12)
    ax2.set_title('Gradient Norm (High values indicate instability)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_dynamics.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/training_dynamics.pdf")
    
    # 4. Advantages
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, metrics['advantages_mean'], 'o-', label='Mean', linewidth=2, markersize=6)
    ax.plot(steps, metrics['advantages_max'], '^-', label='Max', 
            linewidth=2, markersize=6, alpha=0.7)
    ax.plot(steps, metrics['advantages_min'], 'v-', label='Min', 
            linewidth=2, markersize=6, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Advantage', fontsize=12)
    ax.set_title('Advantage Estimates', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/advantages.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/advantages.pdf")
    
    # 5. Response Length
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, metrics['response_length_mean'], 'o-', 
            color='teal', linewidth=2, markersize=6)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Response Length (tokens)', fontsize=12)
    ax.set_title('Mean Response Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/response_length.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/response_length.pdf")
    
    # 6. Summary Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps, reward_mean, 'o-', label='Train Reward', linewidth=2, markersize=5, color='blue')
    if np.any(valid_std_mask):
        ax1.fill_between(steps[valid_std_mask], 
                        (reward_mean - reward_std)[valid_std_mask], 
                        (reward_mean + reward_std)[valid_std_mask], 
                        alpha=0.2, color='blue')
    if val_rewards:
        ax1.plot(val_steps, val_rewards, 's-', label='Val Reward', linewidth=2, markersize=7, color='red')
    ax1.set_ylabel('Reward', fontsize=11)
    ax1.set_title('Rewards Overview', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(steps, metrics['kl_loss'], 'o-', linewidth=2, markersize=5)
    ax2.axhline(y=0.02, color='r', linestyle='--', alpha=0.5, label='Target (0.02)')
    ax2.set_ylabel('KL Loss', fontsize=11)
    ax2.set_title('KL Divergence', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(steps, metrics['entropy'], 'o-', color='green', linewidth=2, markersize=5)
    ax3.set_ylabel('Entropy', fontsize=11)
    ax3.set_title('Policy Entropy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(steps, metrics['grad_norm'], 'o-', color='red', linewidth=2, markersize=5)
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Grad Norm', fontsize=11)
    ax4.set_title('Gradient Norm', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(steps, metrics['response_length_mean'], 'o-', 
             color='teal', linewidth=2, markersize=5)
    ax5.set_xlabel('Step', fontsize=11)
    ax5.set_ylabel('Tokens', fontsize=11)
    ax5.set_title('Response Length', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    fig.suptitle('Key Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(f'{save_dir}/dashboard.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/dashboard.pdf")
    
    plt.close('all')

def print_diagnostics(metrics):
    """Print diagnostic information about training."""
    
    print("\n" + "="*60)
    print("TRAINING DIAGNOSTICS")
    print("="*60)
    
    # Get non-nan values
    rewards = [r for r in metrics['reward_mean'] if not np.isnan(r)]
    kl_losses = [k for k in metrics['kl_loss'] if not np.isnan(k)]
    entropies = [e for e in metrics['entropy'] if not np.isnan(e)]
    
    if not rewards:
        print("\nERROR: No reward data found. Check log format.")
        print("Expected: critic/rewards/mean:0.123")
        return
    
    print(f"\n📊 REWARD PROGRESS:")
    print(f"   Initial: {rewards[0]:.4f}")
    print(f"   Final:   {rewards[-1]:.4f}")
    print(f"   Change:  {rewards[-1] - rewards[0]:.4f}")
    print(f"   Max:     {max(rewards):.4f}")
    
    if not kl_losses:
        print("\nWARNING: No KL loss data found")
        return
    
    print(f"\n🔥 KL DIVERGENCE:")
    avg_kl = np.mean(kl_losses)
    print(f"   Average:  {avg_kl:.6f}")
    print(f"   Range:    {min(kl_losses):.6f} - {max(kl_losses):.6f}")
    if avg_kl < 0.001:
        print(f"   WARNING: KL too low ({avg_kl:.6f}) - increase kl_coef")
    elif avg_kl > 0.1:
        print(f"   WARNING: KL too high ({avg_kl:.6f}) - reduce learning rate or adjust kl_coef")
    else:
        print(f"   KL in acceptable range")
    
    if not entropies:
        print("\nWARNING: No entropy data found")
        return
    
    print(f"\n💫 ENTROPY:")
    print(f"   Initial: {entropies[0]:.4f}")
    print(f"   Final:   {entropies[-1]:.4f}")
    print(f"   Change:  {entropies[-1] - entropies[0]:.4f}")
    if entropies[-1] < 0.1:
        print(f"   WARNING: Very low entropy - consider increasing entropy_coeff")
    
    grad_norms = [g for g in metrics['grad_norm'] if not np.isnan(g)]
    if not grad_norms:
        print("\nWARNING: No gradient norm data found")
    else:
        print(f"\nGRADIENT NORM:")
        avg_grad = np.mean(grad_norms)
        print(f"   Average:  {avg_grad:.4f}")
        print(f"   Max:      {max(grad_norms):.4f}")
        if max(grad_norms) > 10:
            print(f"   WARNING: High gradient norms detected - consider gradient clipping")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if avg_kl < 0.001:
        print("1. INCREASE kl_loss_coef: 0.01 → 0.02 (currently too small)")
    if rewards[-1] - rewards[0] < 0.05:
        print("2. INCREASE learning_rate: 1e-6 → 5e-7 or 1e-6 → 3e-7")
    if entropies[-1] < 0.5:
        print("3. ADD entropy_coeff: 0 → 0.01 (prevent mode collapse)")
    if len(rewards) < 20:
        print("4. TRAIN LONGER: increase total_training_steps to 200+")
    
    print("="*60 + "\n")
    
    if not rewards or not kl_losses or not entropies:
        print("Could not extract metrics. Showing first line of file for debugging:")
        with open(log_file, 'r') as f:
            print(f.readline()[:200])
        return

# Main execution
if __name__ == "__main__":
    import sys
    
    log_file = sys.argv[1] if len(sys.argv) > 1 else "metrics.log"
    
    print(f"Parsing log: {log_file}")
    metrics = parse_training_log(log_file)

    print(f"Found {len(metrics['step'])} steps")

    print("Generating plots")
    plot_training_metrics(metrics)

    print_diagnostics(metrics)

    print("All plots saved to 'plots/'")


def log_file():
    """Global variable for diagnostics."""
    return sys.argv[1] if len(sys.argv) > 1 else "metrics.log"