#!/usr/bin/env python3
"""
Multi-Run Performance Analysis for LLM Boids

This script runs multiple simulations with different random seeds
to get statistical performance data for LLM-powered boids.
Requires OpenAI API key for actual LLM calls.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from boids_llm import LLMBoids
import os
from datetime import datetime
import random

class MultiRunLLMBoidsAnalyzer:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.load_config()
        self.results = []
        
        print("🔍 LLM Boids Multi-Run Performance Analyzer")
        
    def load_config(self):
        """Load configuration including random seeds"""
        with open(f"config/{self.config_file}", 'r') as f:
            self.config = json.load(f)
        self.seeds = self.config.get("random_seeds", [12345, 67890, 11111, 22222, 33333])
        
    def run_single_simulation(self, seed, run_number):
        """Run a single LLM boids simulation with given seed"""
        print(f"\n🔄 Running LLM simulation {run_number + 1}/{len(self.seeds)} with seed {seed}")
        
        try:
            # Create LLM boids instance with specific seed in headless mode
            boids = LLMBoids(self.config_file, headless=True)
            
            # Set random seed for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            
            # Generate initial conditions with seed
            boids.generate_initial_conditions(seed)
            
            # Run simulation without GUI (headless)
            result_data = boids.run_headless()
            
            # Structure the result
            result = {
                'seed': seed,
                'run_number': run_number + 1,
                'agent_stats': result_data.get('agent_stats', {}),
                'swarm_metrics': result_data.get('swarm_metrics', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Show progress
            swarm_metrics = result.get('swarm_metrics', {})
            print(f"✅ LLM Simulation {run_number + 1} completed successfully")
            if swarm_metrics:
                print(f"   Cohesion: {swarm_metrics.get('cohesion', 0):.2f}, "
                      f"Separation: {swarm_metrics.get('separation', 0):.2f}, "
                      f"Alignment: {swarm_metrics.get('alignment', 0):.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ LLM Simulation {run_number + 1} failed: {str(e)}")
            print("💡 Note: LLM boids require OpenAI API key to be set")
            return None
    
    def run_all_simulations(self):
        """Run all simulations with different seeds"""
        print(f"🚀 Starting Multi-Run Analysis for LLM Boids")
        print(f"📊 Running {len(self.seeds)} simulations with seeds: {self.seeds}")
        print()
        
        successful_runs = 0
        
        for i, seed in enumerate(self.seeds):
            result = self.run_single_simulation(seed, i)
            if result:
                self.results.append(result)
                successful_runs += 1
            print()
        
        print(f"🎉 Completed {successful_runs} out of {len(self.seeds)} simulations")
        
        if successful_runs > 0:
            self.print_results()
        else:
            print("❌ No successful simulations to analyze")
            print("💡 Make sure OpenAI API key is properly configured")
            
        return successful_runs > 0
    
    def calculate_statistics(self):
        """Calculate statistical summary of all runs"""
        if not self.results:
            return {}
        
        stats = {
            'agent_stats': {},
            'swarm_stats': {},
            'meta': {
                'total_runs': len(self.results),
                'seeds_used': [r['seed'] for r in self.results]
            }
        }
        
        # Analyze agent performance
        all_agent_types = set()
        for result in self.results:
            agent_stats = result.get('agent_stats', {})
            all_agent_types.update(agent_stats.keys())
        
        for agent_type in all_agent_types:
            agent_data = {}
            for result in self.results:
                agent_stats = result.get('agent_stats', {})
                if agent_type in agent_stats:
                    for metric, value in agent_stats[agent_type].items():
                        if metric not in agent_data:
                            agent_data[metric] = []
                        agent_data[metric].append(value)
            
            # Calculate statistics for each metric
            agent_summary = {}
            for metric, values in agent_data.items():
                if values:
                    agent_summary[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            stats['agent_stats'][agent_type] = agent_summary
        
        # Analyze swarm performance
        swarm_data = {}
        for result in self.results:
            swarm_metrics = result.get('swarm_metrics', {})
            for metric, value in swarm_metrics.items():
                if metric not in swarm_data:
                    swarm_data[metric] = []
                swarm_data[metric].append(value)
        
        for metric, values in swarm_data.items():
            if values:
                stats['swarm_stats'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return stats
    
    def print_results(self):
        """Print comprehensive analysis results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        stats = self.calculate_statistics()
        
        print("=" * 80)
        print("MULTI-RUN LLM BOIDS ANALYSIS".center(80))
        print("=" * 80)
        print(f"Simulations completed: {len(self.results)}")
        seeds_used = [r['seed'] for r in self.results]
        print(f"Seeds used: {seeds_used}")
        print()
        
        # Print agent performance
        print("📊 AGENT PERFORMANCE STATISTICS (n={}):".format(len(self.results)))
        print("-" * 70)
        print()
        
        for agent_type, agent_summary in stats.get('agent_stats', {}).items():
            print(f"🤖 {agent_type}:")
            for metric, metric_stats in agent_summary.items():
                metric_name = metric.replace('_', ' ').title()
                print(f"   {metric_name:<20}: {metric_stats['mean']:.3f} ± {metric_stats['std']:.3f} "
                      f"(min: {metric_stats['min']:.3f}, max: {metric_stats['max']:.3f})")
            print()
        
        # Print swarm performance
        print("🌐 SWARM PERFORMANCE STATISTICS (n={}):".format(len(self.results)))
        print("-" * 70)
        
        for metric, metric_stats in stats.get('swarm_stats', {}).items():
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<20}: {metric_stats['mean']:.3f} ± {metric_stats['std']:.3f} "
                  f"(min: {metric_stats['min']:.3f}, max: {metric_stats['max']:.3f})")
        
        # Overall assessment
        overall_fitness_stats = stats.get('swarm_stats', {}).get('overall_fitness')
        if overall_fitness_stats:
            mean_fitness = overall_fitness_stats['mean']
            std_fitness = overall_fitness_stats['std']
            
            print()
            print("💡 PERFORMANCE ASSESSMENT:")
            print("-" * 40)
            print(f"Overall Fitness: {mean_fitness:.3f} ± {std_fitness:.3f}")
            
            if mean_fitness >= 0.8:
                assessment = "EXCELLENT - High performance across all metrics"
                symbol = "🌟"
            elif mean_fitness >= 0.6:
                assessment = "GOOD - Reliable performance with room for improvement"
                symbol = "✅"
            elif mean_fitness >= 0.4:
                assessment = "MODERATE - Acceptable but inconsistent performance"
                symbol = "⚠️"
            else:
                assessment = "POOR - Significant performance issues detected"
                symbol = "❌"
            
            print(f"Assessment: {assessment}")
            
            if std_fitness > 0.15:
                print("⚠️  High variability detected - LLM responses may be inconsistent")
            elif std_fitness < 0.05:
                print("✅ Low variability - consistent LLM performance across runs")
            else:
                print("✅ Moderate variability - acceptable LLM performance range")
    
    def save_results(self):
        """Save results to JSON file in results/llm_boids folder"""
        # Ensure results directory exists
        results_dir = "results/llm_boids"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/multi_run_llm_boids_{timestamp}.json"
        
        output_data = {
            'metadata': {
                'timestamp': timestamp,
                'config_file': self.config_file,
                'seeds_used': [r['seed'] for r in self.results],
                'successful_runs': len(self.results),
                'total_attempted': len(self.seeds),
                'analysis_type': 'LLM_Boids_Multi_Run'
            },
            'statistics': self.calculate_statistics(),
            'raw_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename
    
    def visualize_results(self):
        """Create visualization of LLM results"""
        if len(self.results) < 2:
            print("❌ Need at least 2 results for visualization")
            return
            
        stats = self.calculate_statistics()
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'LLM Boids Multi-Run Analysis (n={len(self.results)})', fontsize=16)
        
        # 1. Swarm Performance Means
        swarm_stats = stats.get('swarm_stats', {})
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
        
        # 2. Agent Response Times
        agent_stats = stats.get('agent_stats', {})
        if agent_stats:
            agent_names = list(agent_stats.keys())
            response_times = []
            response_time_stds = []
            
            for agent in agent_names:
                if 'avg_response_time' in agent_stats[agent]:
                    response_times.append(agent_stats[agent]['avg_response_time']['mean'])
                    response_time_stds.append(agent_stats[agent]['avg_response_time']['std'])
                else:
                    response_times.append(0)
                    response_time_stds.append(0)
            
            bars2 = ax2.bar(agent_names, response_times, yerr=response_time_stds, 
                           capsize=5, alpha=0.7, color='lightgreen')
            ax2.set_title('LLM Agent Response Times')
            ax2.set_ylabel('Time (seconds)')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, time in zip(bars2, response_times):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{time:.3f}s', ha='center', va='bottom')
        
        # 3. Consistency Scores
        if agent_stats:
            consistency_scores = []
            consistency_stds = []
            
            for agent in agent_names:
                if 'consistency_score' in agent_stats[agent]:
                    consistency_scores.append(agent_stats[agent]['consistency_score']['mean'])
                    consistency_stds.append(agent_stats[agent]['consistency_score']['std'])
                else:
                    consistency_scores.append(0)
                    consistency_stds.append(0)
            
            bars3 = ax3.bar(agent_names, consistency_scores, yerr=consistency_stds,
                           capsize=5, alpha=0.7, color='lightyellow')
            ax3.set_title('LLM Agent Consistency')
            ax3.set_ylabel('Consistency Score (0-1)')
            ax3.set_ylim(0, 1.1)
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars3, consistency_scores):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{score:.3f}', ha='center', va='bottom')
        
        # 4. Run-by-Run Overall Fitness
        if swarm_stats and 'overall_fitness' in swarm_stats:
            run_numbers = [r['run_number'] for r in self.results]
            fitness_values = [r['swarm_metrics'].get('overall_fitness', 0) for r in self.results]
            
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
        print(f"📊 Visualization saved to: {viz_filename}")
        plt.close()  # Close the figure to free memory instead of showing it

def main():
    """Main execution function"""
    print("🔍 LLM Boids Multi-Run Performance Analyzer")
    print("⚠️  Note: Requires OpenAI API key to be configured")
    print()
    
    analyzer = MultiRunLLMBoidsAnalyzer()
    
    # Check if user wants to proceed (since it requires API key)
    try:
        proceed = input("🔑 Do you have OpenAI API key configured? (y/n): ").lower().strip()
        if proceed not in ['y', 'yes']:
            print("❌ LLM analysis requires OpenAI API key. Exiting.")
            print("💡 Use comparison_analyzer_fixed.py for simulated comparison")
            return
    except KeyboardInterrupt:
        print("\n❌ Analysis cancelled.")
        return
    
    success = analyzer.run_all_simulations()
    
    if success:
        analyzer.save_results()
        
        # Always create and save visualization
        print("\n📊 Creating and saving visualization...")
        analyzer.visualize_results()
        
        print("\n✅ Analysis completed with saved results and visualization!")
    else:
        print("❌ No successful runs. Check API key configuration.")

if __name__ == "__main__":
    main()
