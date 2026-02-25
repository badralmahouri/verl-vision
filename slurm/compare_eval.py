#!/usr/bin/env python3
"""
Compare baseline vs trained model on crop evaluation.
Matches experiments by expected_bbox and compares IoU scores.
"""

import re
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def parse_crop_rewards(log_file):
    """Parse CROP_REWARD entries from a log file."""
    results = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find all CROP_REWARD JSON entries
    pattern = r'\[CROP_REWARD\]\s*(\{[^}]+\})'
    matches = re.findall(pattern, content)
    
    for match in matches:
        try:
            data = json.loads(match)
            key = tuple(data.get('expected_bbox', []))
            if key and len(key) == 4:
                results[key] = {
                    'score': data.get('score', 0),
                    'iou': data.get('iou', 0),
                    'pred_bbox': data.get('pred_bbox', []),
                    'expected_bbox': data.get('expected_bbox', []),
                    'used_zoom_tool': data.get('used_zoom_tool', False),
                    'num_tool_calls': data.get('num_tool_calls', 0),
                }
        except json.JSONDecodeError:
            continue
    
    return results


def compute_statistics(baseline_results, trained_results):
    """Compute comparison statistics."""
    # Find matching experiments
    common_keys = set(baseline_results.keys()) & set(trained_results.keys())

    print(f"Baseline experiments: {len(baseline_results)}")
    print(f"Trained experiments:  {len(trained_results)}")
    print(f"Matched experiments:  {len(common_keys)}")
    
    if not common_keys:
        print("ERROR: No matching experiments found!")
        return None
    
    # Extract paired scores
    baseline_scores = []
    trained_scores = []
    improvements = []
    
    for key in common_keys:
        b_score = baseline_results[key]['iou']
        t_score = trained_results[key]['iou']
        baseline_scores.append(b_score)
        trained_scores.append(t_score)
        improvements.append(t_score - b_score)
    
    baseline_scores = np.array(baseline_scores)
    trained_scores = np.array(trained_scores)
    improvements = np.array(improvements)
    
    # Statistics
    stats = {
        'n_samples': len(common_keys),
        'baseline_mean': np.mean(baseline_scores),
        'baseline_std': np.std(baseline_scores),
        'baseline_median': np.median(baseline_scores),
        'trained_mean': np.mean(trained_scores),
        'trained_std': np.std(trained_scores),
        'trained_median': np.median(trained_scores),
        'improvement_mean': np.mean(improvements),
        'improvement_std': np.std(improvements),
        'improved_count': np.sum(improvements > 0),
        'degraded_count': np.sum(improvements < 0),
        'unchanged_count': np.sum(improvements == 0),
        'baseline_zero_count': np.sum(baseline_scores == 0),
        'trained_zero_count': np.sum(trained_scores == 0),
    }
    
    # Percentiles
    for p in [25, 50, 75, 90]:
        stats[f'baseline_p{p}'] = np.percentile(baseline_scores, p)
        stats[f'trained_p{p}'] = np.percentile(trained_scores, p)
    
    print(f"\n{'='*60}")
    print(f"IoU SCORE COMPARISON")
    print(f"{'='*60}")
    print(f"                    Baseline    Trained     Change")
    print(f"  Mean:             {stats['baseline_mean']:.4f}      {stats['trained_mean']:.4f}      {stats['improvement_mean']:+.4f}")
    print(f"  Std:              {stats['baseline_std']:.4f}      {stats['trained_std']:.4f}")
    print(f"  Median:           {stats['baseline_median']:.4f}      {stats['trained_median']:.4f}")
    print(f"  25th percentile:  {stats['baseline_p25']:.4f}      {stats['trained_p25']:.4f}")
    print(f"  75th percentile:  {stats['baseline_p75']:.4f}      {stats['trained_p75']:.4f}")
    print(f"  90th percentile:  {stats['baseline_p90']:.4f}      {stats['trained_p90']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"IMPROVEMENT BREAKDOWN")
    print(f"{'='*60}")
    print(f"  Improved (trained > baseline): {stats['improved_count']} ({100*stats['improved_count']/stats['n_samples']:.1f}%)")
    print(f"  Degraded (trained < baseline): {stats['degraded_count']} ({100*stats['degraded_count']/stats['n_samples']:.1f}%)")
    print(f"  Unchanged:                     {stats['unchanged_count']} ({100*stats['unchanged_count']/stats['n_samples']:.1f}%)")
    print(f"\n  Zero IoU (complete miss):")
    print(f"    Baseline: {stats['baseline_zero_count']} ({100*stats['baseline_zero_count']/stats['n_samples']:.1f}%)")
    print(f"    Trained:  {stats['trained_zero_count']} ({100*stats['trained_zero_count']/stats['n_samples']:.1f}%)")
    
    # Statistical significance
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_rel(trained_scores, baseline_scores)
    wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(trained_scores, baseline_scores, alternative='two-sided')
    
    print(f"\n{'='*60}")
    print(f"STATISTICAL SIGNIFICANCE")
    print(f"{'='*60}")
    print(f"  Paired t-test:     t={t_stat:.4f}, p={p_value:.6f}")
    print(f"  Wilcoxon signed:   W={wilcoxon_stat:.4f}, p={wilcoxon_p:.6f}")
    if p_value < 0.05:
        direction = "BETTER" if stats['improvement_mean'] > 0 else "WORSE"
        print(f"  → Trained model is significantly {direction} (p < 0.05)")
    else:
        print(f"  → No significant difference (p >= 0.05)")
    
    return {
        'baseline_scores': baseline_scores,
        'trained_scores': trained_scores,
        'improvements': improvements,
        'stats': stats,
        'common_keys': common_keys,
        'baseline_results': baseline_results,
        'trained_results': trained_results,
    }


def create_plots(data, output_dir='plots'):
    """Generate comparison plots."""
    Path(output_dir).mkdir(exist_ok=True)
    
    baseline_scores = data['baseline_scores']
    trained_scores = data['trained_scores']
    improvements = data['improvements']
    stats = data['stats']
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Scatter plot: baseline vs trained
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(baseline_scores, trained_scores, alpha=0.5, s=50, c='blue', edgecolors='black', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (no change)')
    ax.set_xlabel('Baseline IoU', fontsize=14)
    ax.set_ylabel('Trained IoU', fontsize=14)
    ax.set_title('Baseline vs Trained Model IoU Scores', fontsize=16, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=12)
    ax.set_aspect('equal')
    
    # Add annotations
    improved = np.sum(trained_scores > baseline_scores)
    degraded = np.sum(trained_scores < baseline_scores)
    ax.text(0.05, 0.95, f'Improved: {improved}', transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', color='green', fontweight='bold')
    ax.text(0.05, 0.90, f'Degraded: {degraded}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/scatter_comparison.png")
    
    # 2. Histogram of improvements
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(-1, 1, 41)
    ax.hist(improvements, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(improvements):.4f}')
    ax.set_xlabel('IoU Improvement (Trained - Baseline)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Distribution of IoU Improvements', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_histogram.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/improvement_histogram.png")
    
    # 3. Box plot comparison
    fig, ax = plt.subplots(figsize=(8, 8))
    bp = ax.boxplot([baseline_scores, trained_scores], labels=['Baseline', 'Trained'],
                    patch_artist=True, widths=0.6)
    colors = ['lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # Make median lines more visible
    for median in bp['medians']:
        median.set_color('darkblue')
        median.set_linewidth(2)
    ax.set_ylabel('IoU Score', fontsize=14)
    ax.set_title('IoU Score Distribution Comparison', fontsize=16, fontweight='bold')
    
    # Add mean markers
    means = [np.mean(baseline_scores), np.mean(trained_scores)]
    ax.scatter([1, 2], means, color='black', s=100, zorder=5, marker='D', label='Mean')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplot_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/boxplot_comparison.png")
    
    # 4. CDF plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_baseline = np.sort(baseline_scores)
    sorted_trained = np.sort(trained_scores)
    cdf = np.arange(1, len(sorted_baseline) + 1) / len(sorted_baseline)
    
    ax.plot(sorted_baseline, cdf, 'r-', linewidth=2, label=f'Baseline (mean={np.mean(baseline_scores):.3f})')
    ax.plot(sorted_trained, cdf, 'g-', linewidth=2, label=f'Trained (mean={np.mean(trained_scores):.3f})')
    ax.set_xlabel('IoU Score', fontsize=14)
    ax.set_ylabel('Cumulative Probability', fontsize=14)
    ax.set_title('Cumulative Distribution of IoU Scores', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cdf_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/cdf_comparison.png")
    
    # 5. Improvement by baseline score range
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', 
                  '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    
    bin_improvements = []
    bin_counts = []
    for i in range(len(bins) - 1):
        mask = (baseline_scores >= bins[i]) & (baseline_scores < bins[i+1])
        if np.sum(mask) > 0:
            bin_improvements.append(np.mean(improvements[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_improvements.append(0)
            bin_counts.append(0)
    
    x = np.arange(len(bin_labels))
    bars = ax.bar(x, bin_improvements, color=['green' if v > 0 else 'red' for v in bin_improvements],
                  alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_xlabel('Baseline IoU Range', fontsize=14)
    ax.set_ylabel('Mean Improvement', fontsize=14)
    ax.set_title('Average Improvement by Baseline Performance', fontsize=16, fontweight='bold')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'n={count}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_by_baseline.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/improvement_by_baseline.png")
    
    # 6. Summary dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(baseline_scores, trained_scores, alpha=0.4, s=30, c='blue')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=1.5)
    ax1.set_xlabel('Baseline IoU', fontsize=11)
    ax1.set_ylabel('Trained IoU', fontsize=11)
    ax1.set_title('Baseline vs Trained', fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    
    # Histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(improvements, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax2.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=1.5)
    ax2.set_xlabel('Improvement', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Improvement Distribution', fontsize=12, fontweight='bold')
    
    # Box plot
    ax3 = fig.add_subplot(gs[0, 2])
    bp = ax3.boxplot([baseline_scores, trained_scores], labels=['Baseline', 'Trained'],
                     patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    # Make median lines more visible
    for median in bp['medians']:
        median.set_color('darkblue')
        median.set_linewidth(2)
    ax3.set_ylabel('IoU Score', fontsize=11)
    ax3.set_title('Score Distribution', fontsize=12, fontweight='bold')
    
    # Statistics text
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    text = f"""
    CROP EVALUATION COMPARISON SUMMARY
    {'='*50}
    
    Samples Compared: {stats['n_samples']}
    
    MEAN IoU:
      Baseline: {stats['baseline_mean']:.4f} (±{stats['baseline_std']:.4f})
      Trained:  {stats['trained_mean']:.4f} (±{stats['trained_std']:.4f})
      Change:   {stats['improvement_mean']:+.4f}
    
    BREAKDOWN:
      Improved: {stats['improved_count']} ({100*stats['improved_count']/stats['n_samples']:.1f}%)
      Degraded: {stats['degraded_count']} ({100*stats['degraded_count']/stats['n_samples']:.1f}%)
      Same:     {stats['unchanged_count']} ({100*stats['unchanged_count']/stats['n_samples']:.1f}%)
    
    ZERO IoU (Complete Miss):
      Baseline: {stats['baseline_zero_count']} ({100*stats['baseline_zero_count']/stats['n_samples']:.1f}%)
      Trained:  {stats['trained_zero_count']} ({100*stats['trained_zero_count']/stats['n_samples']:.1f}%)
    """
    
    ax4.text(0.5, 0.5, text, transform=ax4.transAxes, fontsize=12, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Baseline vs Trained Model - Crop Evaluation', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(f'{output_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/summary_dashboard.png")
    
    plt.close('all')


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_eval.py <baseline_log> <trained_log> [output_dir]")
        print("\nExample:")
        print("  python compare_eval.py crop_eval_baseline.err crop_eval_trained.err plots")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    trained_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'plots'
    
    print(f"Baseline log: {baseline_file}")
    print(f"Trained log:  {trained_file}")
    print(f"Output dir:   {output_dir}")
    
    # Parse logs
    print("\nParsing baseline results...")
    baseline_results = parse_crop_rewards(baseline_file)
    print(f"  Found {len(baseline_results)} experiments")
    
    print("Parsing trained results...")
    trained_results = parse_crop_rewards(trained_file)
    print(f"  Found {len(trained_results)} experiments")
    
    # Compute statistics
    data = compute_statistics(baseline_results, trained_results)
    
    if data:
        # Generate plots
        print(f"\nGenerating plots...")
        create_plots(data, output_dir)
        
        print(f"All plots saved to: {output_dir}/")

if __name__ == "__main__":
    main()
