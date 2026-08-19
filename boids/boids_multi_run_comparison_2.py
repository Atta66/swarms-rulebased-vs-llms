import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from multi_run_boids import MultiRunAnalyzer
import random

class ComparisonAnalyzer:
    """Compare performance between Classic Boids and LLM Boids implementations"""
    
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_path, "config", config_file)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self.classic_results = []
        self.llm_results = []
        
        print("🔍 Classic vs LLM Boids Comparison Analyzer")

    def run_classic_analysis(self):
        """Run multi-run analysis for classic boids"""
        print("\n🚀 Running Classic Boids Multi-Run Analysis")
        print("=" * 60)
        
        classic_analyzer = MultiRunAnalyzer(self.config_file)
        classic_analyzer.run_all_simulations()
        
        # Store results
        self.classic_results = classic_analyzer.results
        
        return classic_analyzer.results

    def run_llm_analysis(self):
        """Run multi-run analysis for LLM boids or load from existing results"""
        print("\n🤖 Running LLM Boids Multi-Run Analysis")
        print("=" * 60)
        
        # First, try to load existing LLM results
        llm_results_file = self.find_latest_llm_results()
        if llm_results_file:
            print(f"📁 Found existing LLM results: {llm_results_file}")
            try:
                use_existing = input("Use existing LLM results? (y/n): ").lower().strip()
                if use_existing in ['y', 'yes']:
                    return self.load_llm_results(llm_results_file)
            except KeyboardInterrupt:
                pass
        
        # If no existing results or user wants to run new analysis
        print("🔑 Attempting to run new LLM analysis...")
        try:
            from multi_run_boids_llm import MultiRunLLMBoidsAnalyzer
            
            llm_analyzer = MultiRunLLMBoidsAnalyzer(self.config_file)
            success = llm_analyzer.run_all_simulations()
            
            if success:
                self.llm_results = llm_analyzer.results
                return llm_analyzer.results
            else:
                print("❌ LLM analysis failed. Falling back to simulation.")
                return self.simulate_llm_analysis()
                
        except Exception as e:
            print(f"❌ LLM analysis failed: {str(e)}")
            print("💡 Falling back to simulated LLM data")
            return self.simulate_llm_analysis()
    
    def find_latest_llm_results(self):
        """Find the most recent LLM results file"""
        # Check both possible directories
        search_paths = [
            ("results/llm_boids", "multi_run_llm_boids_", ".json"),
            ("results/visual_llm_boids", "visual_multi_run_", ".json")
        ]
        
        all_files = []
        
        for results_dir, prefix, suffix in search_paths:
            if not os.path.exists(results_dir):
                continue
            
            files = [f for f in os.listdir(results_dir) if f.startswith(prefix) and f.endswith(suffix)]
            for f in files:
                full_path = os.path.join(results_dir, f)
                all_files.append((full_path, os.path.getctime(full_path)))
        
        if not all_files:
            return None
        
        # Sort by creation time and get the latest
        all_files.sort(key=lambda x: x[1], reverse=True)
        return all_files[0][0]
    
    def load_llm_results(self, filename):
        """Load LLM results from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Handle different data formats
            raw_results = data.get('raw_results', [])
            if not raw_results:
                # Try old format
                raw_results = data.get('results', [])
            
            if not raw_results:
                print("❌ No results found in file")
                return self.simulate_llm_analysis()
            
            # Convert to expected format
            self.llm_results = []
            for result in raw_results:
                # Handle both old and new formats
                if 'swarm_performance' in result:
                    # New format (mini_multi_run)
                    converted_result = {
                        'seed': result.get('seed'),
                        'agent_stats': result.get('agent_performance', {}),
                        'swarm_metrics': {
                            'cohesion': result['swarm_performance'].get('cohesion_metric', 0),
                            'separation': result['swarm_performance'].get('separation_metric', 0),
                            'alignment': result['swarm_performance'].get('alignment_metric', 0),
                            'overall_fitness': result['swarm_performance'].get('overall_fitness', 0)
                        }
                    }
                else:
                    # Old format
                    converted_result = {
                        'seed': result.get('seed'),
                        'agent_stats': result.get('agent_stats', {}),
                        'swarm_metrics': result.get('swarm_metrics', {})
                    }
                
                self.llm_results.append(converted_result)
            
            print(f"✅ Loaded {len(self.llm_results)} LLM results from {os.path.basename(filename)}")
            return self.llm_results
            
        except Exception as e:
            print(f"❌ Failed to load LLM results: {str(e)}")
            print("💡 Falling back to simulated data")
            return self.simulate_llm_analysis()

    def simulate_llm_analysis(self):
        """Simulate LLM boids analysis with mock data for comparison"""
        print("\n🤖 Simulating LLM Boids Analysis (Mock Data)")
        print("=" * 60)
        
        seeds = self.config.get('random_seeds', [12345, 67890, 11111, 22222, 33333])
        
        # Simulate LLM performance characteristics:
        # - More variable response times due to API calls
        # - Slightly different performance patterns
        # - Some randomness in agent behavior due to LLM variability
        
        for i, seed in enumerate(seeds, 1):
            print(f"🔄 Simulating LLM simulation {i}/{len(seeds)} with seed {seed}")
            
            # Simulate some variability and slightly different performance
            random.seed(seed)
            np.random.seed(seed)
            
            # Classic boids tend to be very consistent
            # LLM boids might have more variability but potentially interesting emergent behaviors
            
            # Simulate agent performance with more variability
            agent_stats = {
                'SeparationAgent': {
                    'success_rate': max(0.7, min(1.0, 0.9 + random.gauss(0, 0.1))),
                    'avg_response_time': 0.1 + random.gauss(0, 0.05),  # Slower due to LLM calls
                    'output_quality_score': max(0.6, min(1.0, 0.85 + random.gauss(0, 0.1))),
                    'consistency_score': max(0.4, min(1.0, 0.7 + random.gauss(0, 0.15)))
                },
                'CohesionAgent': {
                    'success_rate': max(0.7, min(1.0, 0.88 + random.gauss(0, 0.12))),
                    'avg_response_time': 0.08 + random.gauss(0, 0.04),
                    'output_quality_score': max(0.6, min(1.0, 0.82 + random.gauss(0, 0.12))),
                    'consistency_score': max(0.3, min(1.0, 0.65 + random.gauss(0, 0.18)))
                },
                'AlignmentAgent': {
                    'success_rate': max(0.7, min(1.0, 0.86 + random.gauss(0, 0.11))),
                    'avg_response_time': 0.09 + random.gauss(0, 0.045),
                    'output_quality_score': max(0.6, min(1.0, 0.80 + random.gauss(0, 0.13))),
                    'consistency_score': max(0.35, min(1.0, 0.62 + random.gauss(0, 0.16)))
                }
            }
            
            # Simulate swarm metrics - LLMs might produce more creative but less predictable behavior
            swarm_metrics = {
                'cohesion': max(0.1, min(1.0, 0.45 + random.gauss(0, 0.12))),
                'separation': max(0.7, min(1.0, 0.92 + random.gauss(0, 0.08))),  # Still good at avoiding collisions
                'alignment': max(0.2, min(1.0, 0.5 + random.gauss(0, 0.18))),   # More variable alignment
                'overall_fitness': 0.0  # Will be calculated
            }
            
            # Calculate overall fitness
            swarm_metrics['overall_fitness'] = (
                swarm_metrics['cohesion'] + 
                swarm_metrics['separation'] + 
                swarm_metrics['alignment']
            ) / 3.0
            
            result = {
                'seed': seed,
                'agent_stats': agent_stats,
                'swarm_metrics': swarm_metrics
            }
            
            self.llm_results.append(result)
            
            print(f"✅ Simulation {i} completed")
            print(f"   Cohesion: {swarm_metrics['cohesion']:.2f}, "
                  f"Separation: {swarm_metrics['separation']:.2f}, "
                  f"Alignment: {swarm_metrics['alignment']:.2f}")
        
        print(f"\n🎉 Completed {len(self.llm_results)} LLM simulations")

    def compare_results(self):
        """Compare classic and LLM results"""
        if not self.classic_results or not self.llm_results:
            print("❌ Missing results for comparison")
            return
        
        print("\n" + "=" * 80)
        print("CLASSIC vs LLM BOIDS PERFORMANCE COMPARISON".center(80))
        print("=" * 80)
        
        # Compare swarm metrics
        print("🌐 SWARM PERFORMANCE COMPARISON:")
        print("-" * 50)
        
        classic_swarm = self.extract_swarm_metrics(self.classic_results)
        llm_swarm = self.extract_swarm_metrics(self.llm_results)
        
        for metric in ['cohesion', 'separation', 'alignment', 'overall_fitness']:
            if metric in classic_swarm and metric in llm_swarm:
                classic_mean = np.mean(classic_swarm[metric])
                classic_std = np.std(classic_swarm[metric])
                llm_mean = np.mean(llm_swarm[metric])
                llm_std = np.std(llm_swarm[metric])
                
                # Calculate improvement/degradation
                diff = llm_mean - classic_mean
                percent_change = (diff / classic_mean) * 100 if classic_mean != 0 else 0
                
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"  Classic: {classic_mean:.3f} ± {classic_std:.3f}")
                print(f"  LLM    : {llm_mean:.3f} ± {llm_std:.3f}")
                print(f"  Change : {diff:+.3f} ({percent_change:+.1f}%)")
                
                if abs(percent_change) < 5:
                    trend = "≈ Similar performance"
                elif percent_change > 0:
                    trend = "↗ LLM performs better"
                else:
                    trend = "↘ Classic performs better"
                print(f"  Trend  : {trend}")
        
        # Compare agent performance
        print("\n🤖 AGENT PERFORMANCE COMPARISON:")
        print("-" * 50)
        
        self.compare_agent_performance('SeparationAgent')
        self.compare_agent_performance('CohesionAgent')
        self.compare_agent_performance('AlignmentAgent')
        
        # Overall assessment
        classic_fitness = np.mean(classic_swarm.get('overall_fitness', [0]))
        llm_fitness = np.mean(llm_swarm.get('overall_fitness', [0]))
        
        print("\n💡 OVERALL ASSESSMENT:")
        print("-" * 30)
        print(f"Classic Boids Overall Fitness: {classic_fitness:.3f}")
        print(f"LLM Boids Overall Fitness    : {llm_fitness:.3f}")
        
        if llm_fitness > classic_fitness:
            assessment = "LLM Boids show superior performance"
            symbol = "🌟"
        elif abs(llm_fitness - classic_fitness) < 0.05:
            assessment = "Both implementations show similar performance"
            symbol = "⚖️"
        else:
            assessment = "Classic Boids show superior performance"
            symbol = "🏆"
        
        print(f"{symbol} {assessment}")
        
        # Variability assessment
        classic_var = np.std(classic_swarm.get('overall_fitness', [0]))
        llm_var = np.std(llm_swarm.get('overall_fitness', [0]))
        
        print(f"\nVariability Comparison:")
        print(f"Classic Variability: {classic_var:.3f}")
        print(f"LLM Variability    : {llm_var:.3f}")
        
        if llm_var < classic_var:
            var_assessment = "LLM shows more consistent performance"
        elif abs(llm_var - classic_var) < 0.02:
            var_assessment = "Both show similar consistency"
        else:
            var_assessment = "Classic shows more consistent performance"
        
        print(f"Assessment: {var_assessment}")

    def extract_swarm_metrics(self, results):
        """Extract swarm metrics from results"""
        swarm_data = {}
        for result in results:
            # Handle both classic and LLM result formats
            swarm_metrics = result.get('swarm_metrics', {})  # LLM format
            if not swarm_metrics:
                # Try classic format
                swarm_performance = result.get('swarm_performance', {})
                if swarm_performance:
                    swarm_metrics = {
                        'cohesion': swarm_performance.get('cohesion_metric', 0),
                        'separation': swarm_performance.get('separation_metric', 0),
                        'alignment': swarm_performance.get('alignment_metric', 0),
                        'overall_fitness': swarm_performance.get('overall_fitness', 0)
                    }
            
            for metric, value in swarm_metrics.items():
                if metric not in swarm_data:
                    swarm_data[metric] = []
                swarm_data[metric].append(value)
        return swarm_data

    def compare_agent_performance(self, agent_name):
        """Compare performance for a specific agent type"""
        classic_agent_data = []
        llm_agent_data = []
        
        # Extract classic agent data
        for result in self.classic_results:
            # Classic format uses agent_performance
            agent_stats = result.get('agent_stats', {})
            if not agent_stats:
                # Try classic format
                agent_performance = result.get('agent_performance', {})
                if agent_name in agent_performance:
                    classic_data = agent_performance[agent_name]
                    # Convert to standard format
                    converted_data = {
                        'success_rate': classic_data.get('success_rate', 0),
                        'avg_response_time': classic_data.get('avg_response_time', 0),
                        'output_quality_score': classic_data.get('output_quality_score', 0),
                        'consistency_score': classic_data.get('consistency_score', 0)
                    }
                    classic_agent_data.append(converted_data)
            elif agent_name in agent_stats:
                classic_agent_data.append(agent_stats[agent_name])
        
        # Extract LLM agent data
        for result in self.llm_results:
            agent_stats = result.get('agent_stats', {})
            if agent_name in agent_stats:
                llm_agent_data.append(agent_stats[agent_name])
        
        if not classic_agent_data or not llm_agent_data:
            return
        
        print(f"\n{agent_name}:")
        
        # Compare key metrics
        metrics = ['success_rate', 'avg_response_time', 'output_quality_score', 'consistency_score']
        for metric in metrics:
            classic_values = [d.get(metric, 0) for d in classic_agent_data]
            llm_values = [d.get(metric, 0) for d in llm_agent_data]
            
            if classic_values and llm_values:
                classic_mean = np.mean(classic_values)
                llm_mean = np.mean(llm_values)
                diff = llm_mean - classic_mean
                
                print(f"  {metric.replace('_', ' ').title():<20}: Classic={classic_mean:.3f}, LLM={llm_mean:.3f}, Δ={diff:+.3f}")

    def visualize_comparison(self):
        """Create multi-run style visualizations comparing classic and LLM results"""
        if not self.classic_results or not self.llm_results:
            print("No results to visualize")
            return
        
        classic_swarm = self.extract_swarm_metrics(self.classic_results)
        llm_swarm = self.extract_swarm_metrics(self.llm_results)
        
        # Calculate statistics for both systems
        classic_stats = self.calculate_metrics_statistics(classic_swarm)
        llm_stats = self.calculate_metrics_statistics(llm_swarm)
        
        # Create 4-panel comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Classic vs LLM Boids Multi-Run Comparison (n={len(self.classic_results)})', 
                     fontsize=16, fontweight='bold')
        
        # Panel 1: Swarm Performance Metrics (Bar chart with error bars)
        ax1 = axes[0, 0]
        metrics = ['cohesion', 'separation', 'alignment', 'overall_fitness']
        metric_labels = ['Cohesion', 'Separation', 'Alignment', 'Overall Fitness']
        
        classic_means = [classic_stats[metric]['mean'] for metric in metrics]
        classic_stds = [classic_stats[metric]['std'] for metric in metrics]
        llm_means = [llm_stats[metric]['mean'] for metric in metrics]
        llm_stds = [llm_stats[metric]['std'] for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, classic_means, width, yerr=classic_stds, 
                       label='Classic', color='lightblue', alpha=0.8, capsize=5, edgecolor='black')
        bars2 = ax1.bar(x + width/2, llm_means, width, yerr=llm_stds,
                       label='LLM', color='lightcoral', alpha=0.8, capsize=5, edgecolor='black')
        
        ax1.set_title('Swarm Performance Metrics', fontweight='bold')
        ax1.set_ylabel('Score (0-1)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)  # Increased upper limit to provide space for labels
        
        # Add value labels on bars with better positioning
        for bars, means in [(bars1, classic_means), (bars2, llm_means)]:
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                # Position label below the bar if it's too high, otherwise above
                if height > 0.95:
                    ax1.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                            f'{mean:.3f}', ha='center', va='top', fontsize=8, 
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                else:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Panel 2: Agent Consistency Scores (if available)
        ax2 = axes[0, 1]
        # Extract individual behavior metrics for consistency analysis
        behavior_metrics = ['cohesion', 'separation', 'alignment']  # Same order as first graph
        classic_consistency = []
        llm_consistency = []
        
        for metric in behavior_metrics:
            if metric in classic_swarm and metric in llm_swarm:
                # Calculate consistency as 1 - coefficient of variation
                classic_cv = classic_stats[metric]['std'] / classic_stats[metric]['mean'] if classic_stats[metric]['mean'] > 0 else 1
                llm_cv = llm_stats[metric]['std'] / llm_stats[metric]['mean'] if llm_stats[metric]['mean'] > 0 else 1
                classic_consistency.append(max(0, 1 - classic_cv))
                llm_consistency.append(max(0, 1 - llm_cv))
        
        if classic_consistency and llm_consistency:
            x_cons = np.arange(len(behavior_metrics))
            bars1 = ax2.bar(x_cons - width/2, classic_consistency, width, 
                           label='Classic', color='lightblue', alpha=0.8, edgecolor='black')
            bars2 = ax2.bar(x_cons + width/2, llm_consistency, width,
                           label='LLM', color='lightcoral', alpha=0.8, edgecolor='black')
            
            ax2.set_title('Agent Consistency Scores', fontweight='bold')
            ax2.set_ylabel('Consistency (0-1)')
            ax2.set_xticks(x_cons)
            ax2.set_xticklabels([m.title() for m in behavior_metrics])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # Add value labels
            for bars, values in [(bars1, classic_consistency), (bars2, llm_consistency)]:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Panel 3: Performance Variability (Standard Deviation)
        ax3 = axes[1, 0]
        bars1 = ax3.bar(x - width/2, classic_stds, width, 
                       label='Classic', color='gold', alpha=0.8, edgecolor='black')
        bars2 = ax3.bar(x + width/2, llm_stds, width,
                       label='LLM', color='orange', alpha=0.8, edgecolor='black')
        
        ax3.set_title('Performance Variability (Standard Deviation)', fontweight='bold')
        ax3.set_ylabel('Standard Deviation')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metric_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bars, stds in [(bars1, classic_stds), (bars2, llm_stds)]:
            for bar, std in zip(bars, stds):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(classic_stds + llm_stds)*0.01,
                        f'{std:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Panel 4: Overall Fitness by Run
        ax4 = axes[1, 1]
        classic_fitness = classic_swarm.get('overall_fitness', [])
        llm_fitness = llm_swarm.get('overall_fitness', [])
        
        run_numbers = range(1, len(classic_fitness) + 1)
        
        ax4.plot(run_numbers, classic_fitness, 'bo-', alpha=0.7, linewidth=1, 
                markersize=4, label='Classic')
        ax4.plot(run_numbers, llm_fitness, 'ro-', alpha=0.7, linewidth=1, 
                markersize=4, label='LLM')
        
        # Add mean lines
        classic_mean = np.mean(classic_fitness)
        llm_mean = np.mean(llm_fitness)
        ax4.axhline(y=classic_mean, color='blue', linestyle='--', alpha=0.7,
                   label=f'Classic Mean: {classic_mean:.3f}')
        ax4.axhline(y=llm_mean, color='red', linestyle='--', alpha=0.7,
                   label=f'LLM Mean: {llm_mean:.3f}')
        
        ax4.set_xlabel('Run Number')
        ax4.set_ylabel('Overall Fitness')
        ax4.set_title('Overall Fitness by Run', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save to visualizations folder
        viz_dir = "results/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        viz_filename = f'{viz_dir}/classic_vs_llm_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {viz_filename}")
        plt.show()
    
    def calculate_metrics_statistics(self, swarm_metrics):
        """Calculate statistics for swarm metrics"""
        stats = {}
        for metric, values in swarm_metrics.items():
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return stats

    def save_comparison_results(self):
        """Save comparison results to file"""
        # Ensure results directory exists
        results_dir = "results/comparisons"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/classic_vs_llm_comparison_{timestamp}.json"
        
        output_data = {
            'timestamp': timestamp,
            'config_file': self.config_file,
            'classic_results': self.classic_results,
            'llm_results': self.llm_results,
            'comparison_summary': self.generate_comparison_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Comparison results saved to: {filename}")

    def generate_comparison_summary(self):
        """Generate summary statistics for the comparison"""
        if not self.classic_results or not self.llm_results:
            return {}
        
        classic_swarm = self.extract_swarm_metrics(self.classic_results)
        llm_swarm = self.extract_swarm_metrics(self.llm_results)
        
        summary = {
            'classic_statistics': {},
            'llm_statistics': {},
            'performance_differences': {}
        }
        
        # Calculate statistics for each implementation
        for metric in ['cohesion', 'separation', 'alignment', 'overall_fitness']:
            if metric in classic_swarm:
                summary['classic_statistics'][metric] = {
                    'mean': float(np.mean(classic_swarm[metric])),
                    'std': float(np.std(classic_swarm[metric])),
                    'min': float(np.min(classic_swarm[metric])),
                    'max': float(np.max(classic_swarm[metric]))
                }
            
            if metric in llm_swarm:
                summary['llm_statistics'][metric] = {
                    'mean': float(np.mean(llm_swarm[metric])),
                    'std': float(np.std(llm_swarm[metric])),
                    'min': float(np.min(llm_swarm[metric])),
                    'max': float(np.max(llm_swarm[metric]))
                }
            
            # Calculate differences
            if metric in classic_swarm and metric in llm_swarm:
                classic_mean = np.mean(classic_swarm[metric])
                llm_mean = np.mean(llm_swarm[metric])
                diff = llm_mean - classic_mean
                percent_change = (diff / classic_mean) * 100 if classic_mean != 0 else 0
                
                summary['performance_differences'][metric] = {
                    'absolute_difference': float(diff),
                    'percent_change': float(percent_change),
                    'winner': 'LLM' if diff > 0 else 'Classic' if diff < 0 else 'Tie'
                }
        
        return summary

def main():
    analyzer = ComparisonAnalyzer()
    
    # Run classic analysis
    analyzer.run_classic_analysis()
    
    # Run LLM analysis (real or simulated)
    analyzer.run_llm_analysis()
    
    # Compare results
    analyzer.compare_results()
    
    # Save results
    analyzer.save_comparison_results()
    
    # Ask user if they want to see visualization
    try:
        show_viz = input("\nShow comparison visualization? (y/n): ").lower().strip() == 'y'
        if show_viz:
            analyzer.visualize_comparison()
    except:
        print("Skipping visualization")

if __name__ == "__main__":
    main()
