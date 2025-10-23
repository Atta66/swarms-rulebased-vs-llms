"""
Individual Boids Performance Visualizer
Creates separate graphs for swarm performance metrics and fitness by trial
Matches ACO visualization styling and colors
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from multi_run_boids import MultiRunAnalyzer
import random

class BoidsIndividualVisualizer:
    """Create individual performance visualizations for Classic vs LLM Boids"""
    
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_path, "config", config_file)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self.classic_results = []
        self.llm_results = []
        
        print("📊 Boids Individual Performance Visualizer")
    
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
                return []
            
            # Convert to expected format
            converted_results = []
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
                
                converted_results.append(converted_result)
            
            print(f"✅ Loaded {len(converted_results)} LLM results from {os.path.basename(filename)}")
            return converted_results
            
        except Exception as e:
            print(f"❌ Failed to load LLM results: {str(e)}")
            return []
    
    def run_classic_analysis(self):
        """Run multi-run analysis for classic boids"""
        print("\n🚀 Running Classic Boids Analysis")
        print("=" * 50)
        
        classic_analyzer = MultiRunAnalyzer(self.config_file)
        classic_analyzer.run_all_simulations()
        
        # Store results
        self.classic_results = classic_analyzer.results
        return classic_analyzer.results
    
    def load_data(self):
        """Load both classic and LLM data"""
        # Run classic analysis
        self.run_classic_analysis()
        
        # Load LLM results
        llm_results_file = self.find_latest_llm_results()
        if llm_results_file:
            print(f"\n📁 Found LLM results: {os.path.basename(llm_results_file)}")
            self.llm_results = self.load_llm_results(llm_results_file)
        else:
            print("❌ No LLM results found")
            self.llm_results = []
    
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
    
    def create_swarm_performance_graph(self):
        """Create swarm performance metrics bar chart"""
        if not self.classic_results:
            print("❌ No classic results available")
            return
        
        classic_swarm = self.extract_swarm_metrics(self.classic_results)
        llm_swarm = self.extract_swarm_metrics(self.llm_results) if self.llm_results else {}
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Metrics to display
        metrics = ['cohesion', 'separation', 'alignment', 'overall_fitness']
        metric_labels = ['Cohesion', 'Separation', 'Alignment', 'Overall Fitness']
        
        # Calculate statistics
        classic_means = []
        classic_stds = []
        llm_means = []
        llm_stds = []
        
        for metric in metrics:
            if metric in classic_swarm:
                classic_means.append(np.mean(classic_swarm[metric]))
                classic_stds.append(np.std(classic_swarm[metric]))
            else:
                classic_means.append(0)
                classic_stds.append(0)
            
            if metric in llm_swarm:
                llm_means.append(np.mean(llm_swarm[metric]))
                llm_stds.append(np.std(llm_swarm[metric]))
            else:
                llm_means.append(0)
                llm_stds.append(0)
        
        # Plot bars - using ACO colors (light blue and light coral/red)
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, classic_means, width, yerr=classic_stds, 
                       label='Classic Boids', color='lightblue', alpha=0.8, capsize=5, 
                       edgecolor='black', linewidth=0.8)
        
        if any(llm_means):  # Only plot LLM bars if we have data
            bars2 = ax.bar(x + width/2, llm_means, width, yerr=llm_stds,
                           label='LLM Boids', color='lightcoral', alpha=0.8, capsize=5,
                           edgecolor='black', linewidth=0.8)
        
        # Styling to match ACO graphs
        ax.set_ylabel('Performance Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars with smart positioning
        for bars, means in [(bars1, classic_means)]:
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                # Position label inside bar if it's too high, otherwise above
                if height > 0.95:
                    ax.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                            f'{mean:.3f}', ha='center', va='top', fontsize=9, 
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        if any(llm_means):
            for bar, mean in zip(bars2, llm_means):
                if mean > 0:  # Only add label if there's actual data
                    height = bar.get_height()
                    # Position label inside bar if it's too high, otherwise above
                    if height > 0.95:
                        ax.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                                f'{mean:.3f}', ha='center', va='top', fontsize=9, 
                                fontweight='bold', color='white',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        viz_dir = "results/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{viz_dir}/boids_swarm_performance_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Swarm performance chart saved: {filename}")
        plt.close()
    
    def create_fitness_by_trial_graph(self):
        """Create overall fitness by trial scatter plot"""
        if not self.classic_results:
            print("❌ No classic results available")
            return
        
        classic_swarm = self.extract_swarm_metrics(self.classic_results)
        llm_swarm = self.extract_swarm_metrics(self.llm_results) if self.llm_results else {}
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get fitness data
        classic_fitness = classic_swarm.get('overall_fitness', [])
        llm_fitness = llm_swarm.get('overall_fitness', [])
        
        # Plot dots only (no lines) - using convergence speed colors (blue and red)
        if classic_fitness:
            classic_trials = range(1, len(classic_fitness) + 1)
            ax.scatter(classic_trials, classic_fitness, color='blue', alpha=0.7, 
                      s=50, label='Classic Boids', edgecolors='darkblue', linewidth=0.5)
        
        if llm_fitness:
            llm_trials = range(1, len(llm_fitness) + 1)
            ax.scatter(llm_trials, llm_fitness, color='red', alpha=0.7, 
                      s=50, label='LLM Boids', edgecolors='darkred', linewidth=0.5)
        
        # Add mean lines (horizontal)
        if classic_fitness:
            classic_mean = np.mean(classic_fitness)
            ax.axhline(y=classic_mean, color='blue', linestyle='-', alpha=0.6, linewidth=2)
        
        if llm_fitness:
            llm_mean = np.mean(llm_fitness)
            ax.axhline(y=llm_mean, color='red', linestyle='-', alpha=0.6, linewidth=2)
        
        # Styling to match ACO convergence speed graph
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('Overall Fitness', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        # Save
        viz_dir = "results/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{viz_dir}/boids_fitness_by_trial_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Fitness by trial chart saved: {filename}")
        plt.close()
    
    def create_all_visualizations(self):
        """Create all individual visualizations"""
        print("\n🎨 Creating Individual Visualizations")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        if not self.classic_results:
            print("❌ No data available for visualization")
            return
        
        # Create individual graphs
        self.create_swarm_performance_graph()
        self.create_fitness_by_trial_graph()
        
        print("\n✅ All individual visualizations created successfully!")

def main():
    visualizer = BoidsIndividualVisualizer()
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()
