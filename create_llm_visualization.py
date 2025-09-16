#!/usr/bin/env python3
"""
Create LLM Boids visualization from mini multi-run results
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob

def find_latest_mini_results():
    """Find the most recent mini multi-run results"""
    results_dir = "results/visual_llm_boids"
    if not os.path.exists(results_dir):
        return None
    
    pattern = os.path.join(results_dir, "visual_multi_run_*.json")
    files = glob(pattern)
    
    if not files:
        return None
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def convert_mini_results_to_stats(data):
    """Convert mini results format to stats format for visualization"""
    results = data.get('raw_results', [])  # Mini script uses 'raw_results'
    
    if not results:
        return None
    
    # Extract swarm metrics from all runs
    swarm_data = {
        'cohesion': [r['swarm_performance']['cohesion_metric'] for r in results],
        'separation': [r['swarm_performance']['separation_metric'] for r in results],
        'alignment': [r['swarm_performance']['alignment_metric'] for r in results],
        'overall_fitness': [r['swarm_performance']['overall_fitness'] for r in results]
    }
    
    # Calculate statistics
    swarm_stats = {}
    for metric, values in swarm_data.items():
        swarm_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Extract agent data (if available in results)
    agent_data = {}
    if results and 'agent_performance' in results[0]:
        # Get all unique agent names
        all_agents = set()
        for r in results:
            all_agents.update(r['agent_performance'].keys())
        
        for agent in all_agents:
            agent_metrics = {
                'success_rate': [],
                'avg_response_time': [],
                'output_quality_score': [],
                'consistency_score': []
            }
            
            for r in results:
                if agent in r['agent_performance']:
                    agent_stats = r['agent_performance'][agent]
                    agent_metrics['success_rate'].append(agent_stats.get('success_rate', 0))
                    agent_metrics['avg_response_time'].append(agent_stats.get('avg_response_time', 0))
                    agent_metrics['output_quality_score'].append(agent_stats.get('output_quality_score', 0))
                    agent_metrics['consistency_score'].append(agent_stats.get('consistency_score', 0))
            
            # Calculate stats for this agent
            agent_data[agent] = {}
            for metric, values in agent_metrics.items():
                if values:  # Only if we have data
                    agent_data[agent][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
    
    return {
        'swarm_stats': swarm_stats,
        'agent_stats': agent_data,
        'raw_results': results
    }

def create_llm_visualization(stats_data, n_runs):
    """Create the 4-panel LLM visualization matching classic boids style"""
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'LLM Boids Multi-Run Analysis (n={n_runs})', fontsize=16)
    
    swarm_stats = stats_data.get('swarm_stats', {})
    agent_stats = stats_data.get('agent_stats', {})
    results = stats_data.get('raw_results', [])
    
    # 1. Swarm Performance Means
    if swarm_stats:
        metrics = ['cohesion', 'separation', 'alignment', 'overall_fitness']
        labels = ['Cohesion', 'Separation', 'Alignment', 'Overall Fitness']
        
        available_metrics = [m for m in metrics if m in swarm_stats]
        available_labels = [labels[metrics.index(m)] for m in available_metrics]
        means = [swarm_stats[m]['mean'] for m in available_metrics]
        stds = [swarm_stats[m]['std'] for m in available_metrics]
        
        bars1 = ax1.bar(available_labels, means, yerr=stds, capsize=5, alpha=0.7, color='lightcoral')
        ax1.set_title('LLM Swarm Performance Metrics')
        ax1.set_ylabel('Score (0-1)')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, mean in zip(bars1, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    # 2. Agent Response Times (or Success Rates if no response time data)
    if agent_stats:
        agent_names = list(agent_stats.keys())
        display_names = [name.replace('Agent', '') for name in agent_names]
        
        # Try response times first, fall back to success rates
        use_response_time = any('avg_response_time' in agent_stats[agent] for agent in agent_names)
        
        if use_response_time:
            values = []
            stds = []
            for agent in agent_names:
                if 'avg_response_time' in agent_stats[agent]:
                    values.append(agent_stats[agent]['avg_response_time']['mean'])
                    stds.append(agent_stats[agent]['avg_response_time']['std'])
                else:
                    values.append(0)
                    stds.append(0)
            
            bars2 = ax2.bar(display_names, values, yerr=stds, 
                           capsize=5, alpha=0.7, color='lightgreen')
            ax2.set_title('LLM Agent Response Times')
            ax2.set_ylabel('Time (seconds)')
            
            for bar, val in zip(bars2, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{val:.3f}s', ha='center', va='bottom')
        else:
            # Use success rates as fallback
            values = []
            stds = []
            for agent in agent_names:
                if 'success_rate' in agent_stats[agent]:
                    values.append(agent_stats[agent]['success_rate']['mean'])
                    stds.append(agent_stats[agent]['success_rate']['std'])
                else:
                    values.append(0)
                    stds.append(0)
            
            bars2 = ax2.bar(display_names, values, yerr=stds, 
                           capsize=5, alpha=0.7, color='lightgreen')
            ax2.set_title('LLM Agent Success Rates')
            ax2.set_ylabel('Success Rate (0-1)')
            ax2.set_ylim(0, 1.1)
            
            for bar, val in zip(bars2, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.3f}', ha='center', va='bottom')
    
    # 3. Performance Variability (Standard Deviations) - like classic boids
    if swarm_stats:
        variability_metrics = ['cohesion', 'separation', 'alignment', 'overall_fitness']
        variability_labels = ['Cohesion', 'Separation', 'Alignment', 'Overall Fitness']
        
        available_metrics = [m for m in variability_metrics if m in swarm_stats]
        available_labels = [variability_labels[variability_metrics.index(m)] for m in available_metrics]
        variability_values = [swarm_stats[m]['std'] for m in available_metrics]
        
        bars3 = ax3.bar(available_labels, variability_values, alpha=0.7, color='gold')
        ax3.set_title('Performance Variability (Standard Deviation)')
        ax3.set_ylabel('Standard Deviation')
        
        for bar, std in zip(bars3, variability_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{std:.3f}', ha='center', va='bottom')
    
    # 4. Run-by-Run Overall Fitness
    if results:
        run_numbers = [i+1 for i in range(len(results))]
        fitness_values = [r['swarm_performance'].get('overall_fitness', 0) for r in results]
        
        ax4.plot(run_numbers, fitness_values, 'o-', linewidth=2, markersize=8, color='purple')
        ax4.set_title('LLM Overall Fitness by Run')
        ax4.set_xlabel('Run Number')
        ax4.set_ylabel('Overall Fitness')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # Add mean line
        mean_fitness = np.mean(fitness_values)
        ax4.axhline(y=mean_fitness, color='red', linestyle='--', 
                   label=f'Mean: {mean_fitness:.3f}')
        ax4.legend()
    
    plt.tight_layout()
    
    # Save visualization
    viz_dir = "results/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    viz_filename = f'{viz_dir}/llm_boids_multi_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    print(f"📊 LLM Boids visualization saved to: {viz_filename}")
    
    # Also show the plot
    plt.show()

def main():
    """Main execution function"""
    print("🎨 Creating LLM Boids Multi-Run Visualization")
    
    # Find the latest mini results
    latest_file = find_latest_mini_results()
    
    if not latest_file:
        print("❌ No mini multi-run results found")
        print("💡 Run: python mini_multi_run_llm.py -n 3 --auto first")
        return
    
    print(f"📂 Using results from: {os.path.basename(latest_file)}")
    
    try:
        # Load the results
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Check if we have results
        if 'raw_results' not in data or not data['raw_results']:
            print("❌ No raw_results found in data file")
            return
        
        n_runs = len(data['raw_results'])
        print(f"📊 Found {n_runs} successful runs")
        
        if n_runs < 2:
            print("❌ Need at least 2 runs for meaningful visualization")
            return
        
        # Convert to stats format
        stats_data = convert_mini_results_to_stats(data)
        
        if not stats_data:
            print("❌ Could not convert results to stats format")
            return
        
        # Create visualization
        create_llm_visualization(stats_data, n_runs)
        print("✅ LLM Boids visualization completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
