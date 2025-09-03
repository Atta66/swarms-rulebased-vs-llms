#!/usr/bin/env python3
"""
LLM vs Classic Boids Performance Comparison Tool

This script compares the performance between LLM-powered boids and classic boids.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

class BoidsComparison:
    def __init__(self, llm_file="performance_data.json", classic_file="classic_boids_performance.json"):
        self.llm_file = llm_file
        self.classic_file = classic_file
        self.llm_data = self.load_data(llm_file)
        self.classic_data = self.load_data(classic_file)
    
    def load_data(self, filename):
        """Load performance data from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return data.get("current_report", {})
        except FileNotFoundError:
            print(f"⚠️  File {filename} not found. Run the simulation first.")
            return {}
    
    def compare_performance(self):
        """Compare LLM vs Classic boid performance"""
        print("\n🔍 LLM vs CLASSIC BOIDS PERFORMANCE COMPARISON")
        print("="*70)
        
        # Get agent data
        llm_agents = self.llm_data.get("agent_performance", {})
        classic_agents = self.classic_data.get("agent_performance", {})
        
        # Create comparison table
        print("\n📊 AGENT PERFORMANCE COMPARISON:")
        print(f"{'Metric':<20} {'LLM Boids':<25} {'Classic Boids':<25} {'Winner':<10}")
        print("-" * 85)
        
        # Compare response times
        llm_avg_time = self._get_average_metric(llm_agents, 'average_response_time')
        classic_avg_time = self._get_average_metric(classic_agents, 'average_response_time')
        
        print(f"{'Response Time':<20} {llm_avg_time:.3f}s {'':<16} {classic_avg_time:.3f}s {'':<16} {'Classic' if classic_avg_time < llm_avg_time else 'LLM':<10}")
        
        # Compare success rates
        llm_success = self._get_average_metric(llm_agents, 'success_rate')
        classic_success = self._get_average_metric(classic_agents, 'success_rate')
        
        print(f"{'Success Rate':<20} {llm_success:.1%} {'':<16} {classic_success:.1%} {'':<16} {'Classic' if classic_success > llm_success else 'LLM':<10}")
        
        # Compare quality scores
        llm_quality = self._get_average_metric(llm_agents, 'output_quality_score')
        classic_quality = self._get_average_metric(classic_agents, 'output_quality_score')
        
        print(f"{'Quality Score':<20} {llm_quality:.2f}/1.0 {'':<13} {classic_quality:.2f}/1.0 {'':<13} {'Classic' if classic_quality > llm_quality else 'LLM':<10}")
        
        # Compare consistency
        llm_consistency = self._get_average_metric(llm_agents, 'consistency_score')
        classic_consistency = self._get_average_metric(classic_agents, 'consistency_score')
        
        print(f"{'Consistency':<20} {llm_consistency:.2f}/1.0 {'':<13} {classic_consistency:.2f}/1.0 {'':<13} {'Classic' if classic_consistency > llm_consistency else 'LLM':<10}")
        
        print(f"\n🌐 SWARM BEHAVIOR COMPARISON:")
        print(f"{'Metric':<20} {'LLM Boids':<25} {'Classic Boids':<25} {'Winner':<10}")
        print("-" * 85)
        
        # Compare swarm metrics
        llm_swarm = self.llm_data.get("swarm_performance", {})
        classic_swarm = self.classic_data.get("swarm_performance", {})
        
        metrics = ['cohesion_metric', 'separation_metric', 'alignment_metric', 'overall_fitness']
        metric_names = ['Cohesion', 'Separation', 'Alignment', 'Overall Fitness']
        
        for metric, name in zip(metrics, metric_names):
            llm_val = llm_swarm.get(metric, 0)
            classic_val = classic_swarm.get(metric, 0)
            winner = 'Classic' if classic_val > llm_val else 'LLM'
            
            print(f"{name:<20} {llm_val:.2f}/1.0 {'':<15} {classic_val:.2f}/1.0 {'':<15} {winner:<10}")
    
    def _get_average_metric(self, agents_data, metric_name):
        """Calculate average metric across all agents"""
        if not agents_data:
            return 0.0
        
        values = [agent.get(metric_name, 0) for agent in agents_data.values()]
        return np.mean(values) if values else 0.0
    
    def generate_insights(self):
        """Generate insights from the comparison"""
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        print("-" * 40)
        
        llm_agents = self.llm_data.get("agent_performance", {})
        classic_agents = self.classic_data.get("agent_performance", {})
        
        # Response time analysis
        llm_time = self._get_average_metric(llm_agents, 'average_response_time')
        classic_time = self._get_average_metric(classic_agents, 'average_response_time')
        
        if llm_time > classic_time:
            speedup = llm_time / classic_time if classic_time > 0 else float('inf')
            print(f"⚡ Classic boids are {speedup:.0f}x faster than LLM boids")
            print(f"   LLM overhead: {(llm_time - classic_time) * 1000:.1f}ms per call")
        
        # Quality analysis
        llm_quality = self._get_average_metric(llm_agents, 'output_quality_score')
        classic_quality = self._get_average_metric(classic_agents, 'output_quality_score')
        
        if classic_quality > llm_quality:
            print(f"🎯 Classic boids have more predictable outputs")
            print(f"   Quality difference: {(classic_quality - llm_quality):.2f}")
        
        # Consistency analysis
        llm_consistency = self._get_average_metric(llm_agents, 'consistency_score')
        classic_consistency = self._get_average_metric(classic_agents, 'consistency_score')
        
        if classic_consistency > llm_consistency:
            print(f"🔄 Classic boids are more consistent")
            print(f"   Consistency advantage: {(classic_consistency - llm_consistency):.2f}")
        
        # Swarm behavior analysis
        llm_swarm = self.llm_data.get("swarm_performance", {})
        classic_swarm = self.classic_data.get("swarm_performance", {})
        
        llm_fitness = llm_swarm.get('overall_fitness', 0)
        classic_fitness = classic_swarm.get('overall_fitness', 0)
        
        if abs(llm_fitness - classic_fitness) > 0.1:
            better = "LLM" if llm_fitness > classic_fitness else "Classic"
            diff = abs(llm_fitness - classic_fitness)
            print(f"🌐 {better} boids show better swarm behavior (+{diff:.2f})")
    
    def visualize_comparison(self):
        """Create comparison visualizations"""
        if not self.llm_data or not self.classic_data:
            print("❌ Missing data for visualization")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LLM vs Classic Boids Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Response Time Comparison
        llm_agents = self.llm_data.get("agent_performance", {})
        classic_agents = self.classic_data.get("agent_performance", {})
        
        llm_times = [agent.get('average_response_time', 0) for agent in llm_agents.values()]
        classic_times = [agent.get('average_response_time', 0) for agent in classic_agents.values()]
        
        x = ['LLM Boids', 'Classic Boids']
        y = [np.mean(llm_times) if llm_times else 0, np.mean(classic_times) if classic_times else 0]
        
        ax1.bar(x, y, color=['red', 'blue'], alpha=0.7)
        ax1.set_title('Average Response Time')
        ax1.set_ylabel('Seconds')
        
        # 2. Success Rate Comparison
        llm_success = [agent.get('success_rate', 0) for agent in llm_agents.values()]
        classic_success = [agent.get('success_rate', 0) for agent in classic_agents.values()]
        
        y = [np.mean(llm_success) * 100 if llm_success else 0, np.mean(classic_success) * 100 if classic_success else 0]
        
        ax2.bar(x, y, color=['red', 'blue'], alpha=0.7)
        ax2.set_title('Success Rate')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_ylim(0, 100)
        
        # 3. Swarm Performance Radar
        llm_swarm = self.llm_data.get("swarm_performance", {})
        classic_swarm = self.classic_data.get("swarm_performance", {})
        
        metrics = ['cohesion_metric', 'separation_metric', 'alignment_metric']
        llm_values = [llm_swarm.get(m, 0) for m in metrics]
        classic_values = [classic_swarm.get(m, 0) for m in metrics]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        llm_values += llm_values[:1]
        classic_values += classic_values[:1]
        angles += angles[:1]
        
        ax3.plot(angles, llm_values, 'o-', linewidth=2, label='LLM Boids', color='red')
        ax3.plot(angles, classic_values, 'o-', linewidth=2, label='Classic Boids', color='blue')
        ax3.fill(angles, llm_values, alpha=0.25, color='red')
        ax3.fill(angles, classic_values, alpha=0.25, color='blue')
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(['Cohesion', 'Separation', 'Alignment'])
        ax3.set_ylim(0, 1)
        ax3.set_title('Swarm Performance Comparison')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Overall Fitness
        llm_fitness = llm_swarm.get('overall_fitness', 0)
        classic_fitness = classic_swarm.get('overall_fitness', 0)
        
        y = [llm_fitness, classic_fitness]
        colors = ['red', 'blue']
        
        ax4.bar(x, y, color=colors, alpha=0.7)
        ax4.set_title('Overall Swarm Fitness')
        ax4.set_ylabel('Fitness Score (0-1)')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main comparison function"""
    print("🔍 LLM vs Classic Boids Performance Comparison Tool")
    
    comparison = BoidsComparison()
    
    if not comparison.llm_data and not comparison.classic_data:
        print("❌ No performance data found. Run both simulations first:")
        print("   1. python boids_llm.py")
        print("   2. python boids.py")
        return
    
    comparison.compare_performance()
    comparison.generate_insights()
    
    try:
        show_viz = input("\nShow comparison charts? (y/n): ").lower().strip()
        if show_viz in ['y', 'yes']:
            comparison.visualize_comparison()
    except KeyboardInterrupt:
        print("\nComparison completed.")

if __name__ == "__main__":
    main()
