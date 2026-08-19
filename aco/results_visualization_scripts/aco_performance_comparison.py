#!/usr/bin/env python3
"""
ACO Performance Comparison Tool
Compares Regular ACO vs LLM-Enhanced ACO performance metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from datetime import datetime

# Add parent directory to path to import from results folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_aco_results():
    """Load the latest ACO results"""
    results_dir = "../results/aco"
    if not os.path.exists(results_dir):
        print(f"❌ No ACO results found in {results_dir}")
        return None
    
    # Find the most recent ACO results file
    aco_files = [f for f in os.listdir(results_dir) if f.startswith("multi_run_aco_") and f.endswith(".json")]
    if not aco_files:
        print("❌ No ACO results files found")
        return None
    
    # Sort by modification time to get the newest
    aco_files_with_time = [(f, os.path.getmtime(os.path.join(results_dir, f))) for f in aco_files]
    latest_file = max(aco_files_with_time, key=lambda x: x[1])[0]
    filepath = os.path.join(results_dir, latest_file)
    print(f"📄 Using ACO results: {latest_file}")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def load_aco_llm_results():
    """Load the latest ACO LLM results"""
    results_dir = "../results/aco_llm"
    if not os.path.exists(results_dir):
        print(f"❌ No ACO LLM results found in {results_dir}")
        return None
    
    # Find the most recent ACO LLM results file
    aco_llm_files = [f for f in os.listdir(results_dir) if f.startswith("multi_run_aco_llm_") and f.endswith(".json")]
    if not aco_llm_files:
        print("❌ No ACO LLM results files found")
        return None
    
    # Sort by modification time to get the newest
    aco_llm_files_with_time = [(f, os.path.getmtime(os.path.join(results_dir, f))) for f in aco_llm_files]
    latest_file = max(aco_llm_files_with_time, key=lambda x: x[1])[0]
    filepath = os.path.join(results_dir, latest_file)
    print(f"📄 Using ACO LLM results: {latest_file}")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def create_comparison_visualization(aco_data, aco_llm_data):
    """Create single mean performance comparison with error bars"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract statistics
    aco_stats = aco_data['statistics']['swarm_stats']
    aco_llm_stats = aco_llm_data['statistics']['swarm_stats']
    
    metrics = ['convergence_speed', 'solution_quality', 'learning_efficiency', 'learning_stability']
    metric_labels = ['Convergence\nSpeed', 'Solution\nQuality', 'Learning\nEfficiency', 'Learning\nStability']
    
    # Extract means and standard deviations for error bars
    aco_means = [aco_stats[metric]['mean'] for metric in metrics]
    aco_stds = [aco_stats[metric]['std'] for metric in metrics]
    aco_llm_means = [aco_llm_stats[metric]['mean'] for metric in metrics]
    aco_llm_stds = [aco_llm_stats[metric]['std'] for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create bars with error bars
    bars1 = ax.bar(x - width/2, aco_means, width, yerr=aco_stds, 
                   label='Classic ACO', color='lightblue', alpha=0.7, 
                   edgecolor='black', capsize=5)
    bars2 = ax.bar(x + width/2, aco_llm_means, width, yerr=aco_llm_stds,
                   label='LLM ACO', color='lightcoral', alpha=0.7, 
                   edgecolor='black', capsize=5)
    
    ax.set_xlabel('Performance Metrics', fontsize=14)
    ax.set_ylabel('Performance Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bars, means in [(bars1, aco_means), (bars2, aco_llm_means)]:
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def print_comparison_summary(aco_data, aco_llm_data):
    """Print detailed comparison summary"""
    print(f"\n{'='*80}")
    print("                  ACO vs LLM-ACO PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    # Basic info
    aco_runs = aco_data['metadata']['successful_runs']
    aco_llm_runs = aco_llm_data['metadata']['successful_runs']
    
    print(f"ACO Successful Runs: {aco_runs}")
    print(f"ACO LLM Successful Runs: {aco_llm_runs}")
    
    # Performance comparison
    print(f"\n📊 Performance Metrics Comparison:")
    print(f"{'Metric':<20} {'ACO Mean':<12} {'ACO LLM Mean':<12} {'Difference':<12} {'Winner':<10}")
    print(f"{'-'*75}")
    
    aco_stats = aco_data['statistics']['swarm_stats']
    aco_llm_stats = aco_llm_data['statistics']['swarm_stats']
    
    metrics = ['convergence_speed', 'solution_quality', 'learning_efficiency', 'learning_stability', 'overall_fitness']
    
    for metric in metrics:
        aco_mean = aco_stats[metric]['mean']
        aco_llm_mean = aco_llm_stats[metric]['mean']
        diff = aco_llm_mean - aco_mean
        winner = "ACO LLM" if diff > 0 else "ACO" if diff < 0 else "Tie"
        
        metric_name = metric.replace('_', ' ').title()
        print(f"{metric_name:<20} {aco_mean:<12.3f} {aco_llm_mean:<12.3f} {diff:<12.3f} {winner:<10}")
    
    # Timing comparison
    print(f"\n⏱️  Timing Comparison:")
    
    # Check if timing stats are available
    if 'timing_stats' in aco_data['statistics'] and 'timing_stats' in aco_llm_data['statistics']:
        aco_timing = aco_data['statistics']['timing_stats']
        aco_llm_timing = aco_llm_data['statistics']['timing_stats']
        
        aco_conv_time = aco_timing['convergence_time']['mean']
        aco_llm_conv_time = aco_llm_timing['convergence_time']['mean']
        time_diff = aco_llm_conv_time - aco_conv_time
        
        print(f"ACO Convergence Time:     {aco_conv_time:.3f}s")
        print(f"ACO LLM Convergence Time: {aco_llm_conv_time:.3f}s")
        print(f"Time Difference:          {time_diff:+.3f}s ({'ACO LLM slower' if time_diff > 0 else 'ACO LLM faster'})")
    else:
        print("⚠️  Timing statistics not available for comparison")
    
    # Enhanced timing comparison (if available)
    if 'overall_timing' in aco_data['statistics'] and 'overall_timing' in aco_llm_data['statistics']:
        aco_overall_timing = aco_data['statistics']['overall_timing']
        aco_llm_overall_timing = aco_llm_data['statistics']['overall_timing']
        
        print(f"\n⏰ Overall Analysis Timing:")
        print(f"ACO Total Time:           {aco_overall_timing['total_duration_minutes']:.2f} min")
        print(f"ACO LLM Total Time:       {aco_llm_overall_timing['total_duration_minutes']:.2f} min")
        print(f"ACO Avg Trial:            {aco_overall_timing['average_trial_duration']:.2f}s")
        print(f"ACO LLM Avg Trial:        {aco_llm_overall_timing['average_trial_duration']:.2f}s")
        print(f"ACO Trials/min:           {aco_overall_timing['trials_per_minute']:.2f}")
        print(f"ACO LLM Trials/min:       {aco_llm_overall_timing['trials_per_minute']:.2f}")
    
    # Performance statistics comparison (if available)
    if 'performance_stats' in aco_data['statistics'] and 'performance_stats' in aco_llm_data['statistics']:
        aco_perf = aco_data['statistics']['performance_stats']
        aco_llm_perf = aco_llm_data['statistics']['performance_stats']
        
        print(f"\n💻 Resource Usage Comparison:")
        print(f"ACO CPU Usage:            {aco_perf['process_cpu_usage']['mean']:.2f}% ± {aco_perf['process_cpu_usage']['std']:.2f}%")
        print(f"ACO LLM CPU Usage:        {aco_llm_perf['process_cpu_usage']['mean']:.2f}% ± {aco_llm_perf['process_cpu_usage']['std']:.2f}%")
        print(f"ACO Memory Usage:         {aco_perf['process_memory_usage_mb']['mean']:.1f}MB ± {aco_perf['process_memory_usage_mb']['std']:.1f}MB")
        print(f"ACO LLM Memory Usage:     {aco_llm_perf['process_memory_usage_mb']['mean']:.1f}MB ± {aco_llm_perf['process_memory_usage_mb']['std']:.1f}MB")
        print(f"ACO Peak Memory:          {aco_perf['max_memory_usage_mb']['max']:.1f}MB")
        print(f"ACO LLM Peak Memory:      {aco_llm_perf['max_memory_usage_mb']['max']:.1f}MB")
        
        if 'gpu_load_percent' in aco_perf and 'gpu_load_percent' in aco_llm_perf:
            print(f"ACO GPU Load:             {aco_perf['gpu_load_percent']['mean']:.1f}% ± {aco_perf['gpu_load_percent']['std']:.1f}%")
            print(f"ACO LLM GPU Load:         {aco_llm_perf['gpu_load_percent']['mean']:.1f}% ± {aco_llm_perf['gpu_load_percent']['std']:.1f}%")
    
    # Overall winner
    aco_overall = aco_stats['overall_fitness']['mean']
    aco_llm_overall = aco_llm_stats['overall_fitness']['mean']
    
    print(f"\n🏆 Overall Performance:")
    print(f"ACO Overall Fitness:      {aco_overall:.3f}")
    print(f"ACO LLM Overall Fitness:  {aco_llm_overall:.3f}")
    
    if aco_llm_overall > aco_overall:
        improvement = ((aco_llm_overall - aco_overall) / aco_overall) * 100
        print(f"🎉 Winner: ACO LLM (+{aco_llm_overall - aco_overall:.3f}, {improvement:+.1f}% improvement)")
    elif aco_overall > aco_llm_overall:
        improvement = ((aco_overall - aco_llm_overall) / aco_llm_overall) * 100
        print(f"🎉 Winner: ACO (+{aco_overall - aco_llm_overall:.3f}, {improvement:+.1f}% advantage)")
    else:
        print(f"🤝 Result: Tie")

def main():
    """Main comparison function"""
    print("🔄 ACO Performance Comparison Tool")
    print("=" * 50)
    
    # Load results
    print("📊 Loading ACO results...")
    aco_data = load_aco_results()
    
    print("🤖 Loading ACO LLM results...")
    aco_llm_data = load_aco_llm_results()
    
    if aco_data is None:
        print("❌ No ACO data available. Please run multi_run_aco.py first.")
        return
    
    if aco_llm_data is None:
        print("❌ No ACO LLM data available. Please run multi_run_aco_llm.py first.")
        print("📝 For now, showing available ACO results...")
        
        # Show ACO summary only
        aco_stats = aco_data['statistics']['swarm_stats']
        print(f"\n📊 ACO Performance Summary:")
        for metric, data in aco_stats.items():
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<25} {data['mean']:.3f} ± {data['std']:.3f}")
        
        return
    
    # Perform comparison
    print_comparison_summary(aco_data, aco_llm_data)
    
    # Create visualization
    print(f"\n📊 Creating comparison visualization...")
    fig = create_comparison_visualization(aco_data, aco_llm_data)
    
    # Save visualization
    viz_dir = "../results/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_filename = f'{viz_dir}/aco_performance_comparison_{timestamp}.png'
    fig.savefig(viz_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison visualization saved to: {viz_filename}")
    
    plt.show()

if __name__ == "__main__":
    main()
