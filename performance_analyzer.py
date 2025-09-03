#!/usr/bin/env python3
"""
Performance Analysis Tool for LLM Swarm Agents

This script provides detailed analysis and visualization of agent performance
including individual agent metrics, swarm behavior analysis, and recommendations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from agent_performance_tracker import AgentPerformanceTracker

# Optional imports for enhanced features
try:
    import seaborn as sns
    import pandas as pd
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False
    print("Note: seaborn and pandas not available. Some features will be limited.")

class PerformanceAnalyzer:
    def __init__(self, performance_file: str = "performance_data.json"):
        self.performance_file = performance_file
        self.data = self.load_data()
        
    def load_data(self):
        """Load performance data from JSON file"""
        try:
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Performance file {self.performance_file} not found.")
            return {}
    
    def analyze_agent_performance(self):
        """Analyze individual agent performance"""
        if not self.data or "current_report" not in self.data:
            print("No performance data available for analysis.")
            return
            
        current_report = self.data["current_report"]
        agent_data = current_report.get("agent_performance", {})
        
        print("\n" + "="*70)
        print("                    DETAILED AGENT ANALYSIS")
        print("="*70)
        
        # Create performance comparison table
        if agent_data:
            print("\n📊 PERFORMANCE COMPARISON TABLE:")
            
            if ENHANCED_FEATURES:
                # Use pandas for better formatting if available
                df_data = []
                for agent_name, metrics in agent_data.items():
                    df_data.append({
                        "Agent": agent_name,
                        "Success Rate (%)": f"{metrics['success_rate']:.1%}",
                        "Avg Response Time (s)": f"{metrics['average_response_time']:.3f}",
                        "Quality Score": f"{metrics['output_quality_score']:.2f}",
                        "Consistency": f"{metrics['consistency_score']:.2f}",
                        "Total Calls": metrics['total_calls'],
                        "Failed Calls": metrics['failed_calls']
                    })
                
                df = pd.DataFrame(df_data)
                print(df.to_string(index=False))
            else:
                # Simple table format without pandas
                print(f"{'Agent':<18} {'Success %':<10} {'Resp Time':<10} {'Quality':<8} {'Consistency':<12} {'Total':<6} {'Failed':<6}")
                print("-" * 80)
                for agent_name, metrics in agent_data.items():
                    print(f"{agent_name:<18} {metrics['success_rate']:<9.1%} "
                          f"{metrics['average_response_time']:<9.3f}s {metrics['output_quality_score']:<7.2f} "
                          f"{metrics['consistency_score']:<11.2f} {metrics['total_calls']:<6} {metrics['failed_calls']:<6}")
            
            
            # Identify best and worst performing agents
            best_agent = max(agent_data.items(), 
                           key=lambda x: x[1]['success_rate'] * x[1]['output_quality_score'])
            worst_agent = min(agent_data.items(), 
                            key=lambda x: x[1]['success_rate'] * x[1]['output_quality_score'])
            
            print(f"\n🏆 BEST PERFORMING AGENT: {best_agent[0]}")
            print(f"   Success Rate: {best_agent[1]['success_rate']:.1%}")
            print(f"   Quality Score: {best_agent[1]['output_quality_score']:.2f}")
            
            print(f"\n⚠️  WORST PERFORMING AGENT: {worst_agent[0]}")
            print(f"   Success Rate: {worst_agent[1]['success_rate']:.1%}")
            print(f"   Quality Score: {worst_agent[1]['output_quality_score']:.2f}")
            
    def analyze_swarm_behavior(self):
        """Analyze overall swarm behavior metrics"""
        if not self.data or "current_report" not in self.data:
            return
            
        swarm_data = self.data["current_report"].get("swarm_performance", {})
        
        print(f"\n🌐 SWARM BEHAVIOR ANALYSIS:")
        print("-" * 40)
        
        cohesion = swarm_data.get("cohesion_metric", 0)
        separation = swarm_data.get("separation_metric", 0)
        alignment = swarm_data.get("alignment_metric", 0)
        fitness = swarm_data.get("overall_fitness", 0)
        
        print(f"Cohesion Metric:    {cohesion:.2f}/1.0  {'✅' if cohesion > 0.7 else '⚠️' if cohesion > 0.4 else '❌'}")
        print(f"Separation Metric:  {separation:.2f}/1.0  {'✅' if separation > 0.7 else '⚠️' if separation > 0.4 else '❌'}")
        print(f"Alignment Metric:   {alignment:.2f}/1.0  {'✅' if alignment > 0.7 else '⚠️' if alignment > 0.4 else '❌'}")
        print(f"Overall Fitness:    {fitness:.2f}/1.0  {'✅' if fitness > 0.7 else '⚠️' if fitness > 0.4 else '❌'}")
        
        # Provide specific feedback
        issues = []
        if cohesion < 0.5:
            issues.append("Poor cohesion - boids are not grouping effectively")
        if separation < 0.5:
            issues.append("Poor separation - boids are colliding too frequently")
        if alignment < 0.5:
            issues.append("Poor alignment - boids are not coordinating velocities")
            
        if issues:
            print(f"\n⚠️  IDENTIFIED ISSUES:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print(f"\n✅ SWARM BEHAVIOR IS OPTIMAL")
    
    def generate_recommendations(self):
        """Generate performance improvement recommendations"""
        if not self.data or "current_report" not in self.data:
            return
            
        current_report = self.data["current_report"]
        agent_data = current_report.get("agent_performance", {})
        swarm_data = current_report.get("swarm_performance", {})
        
        print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")
        print("-" * 50)
        
        recommendations = []
        
        # Agent-specific recommendations
        for agent_name, metrics in agent_data.items():
            if metrics['success_rate'] < 0.7:
                recommendations.append(
                    f"🔧 {agent_name}: Low success rate ({metrics['success_rate']:.1%}). "
                    f"Review prompt clarity and expected output format."
                )
            
            if metrics['average_response_time'] > 2.0:
                recommendations.append(
                    f"⏱️  {agent_name}: Slow response time ({metrics['average_response_time']:.2f}s). "
                    f"Consider simplifying prompts or optimizing model parameters."
                )
            
            if metrics['output_quality_score'] < 0.6:
                recommendations.append(
                    f"📊 {agent_name}: Low quality score ({metrics['output_quality_score']:.2f}). "
                    f"Adjust expected output ranges or improve prompt instructions."
                )
            
            if metrics['consistency_score'] < 0.6:
                recommendations.append(
                    f"🎯 {agent_name}: Inconsistent outputs ({metrics['consistency_score']:.2f}). "
                    f"Add more specific constraints to prompts."
                )
        
        # Swarm-level recommendations
        if swarm_data.get("cohesion_metric", 0) < 0.5:
            recommendations.append(
                "🌐 Improve swarm cohesion by adjusting cohesion weight or perception radius."
            )
        
        if swarm_data.get("separation_metric", 0) < 0.5:
            recommendations.append(
                "🌐 Improve separation by increasing separation weight or adjusting radius parameter."
            )
        
        if swarm_data.get("alignment_metric", 0) < 0.5:
            recommendations.append(
                "🌐 Improve alignment by adjusting alignment weight or velocity constraints."
            )
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i:2d}. {rec}")
        else:
            print("✅ No specific recommendations - system is performing well!")
    
    def visualize_performance(self):
        """Create performance visualization charts"""
        if not self.data or "current_report" not in self.data:
            print("No data available for visualization.")
            return
            
        current_report = self.data["current_report"]
        agent_data = current_report.get("agent_performance", {})
        swarm_data = current_report.get("swarm_performance", {})
        
        if not agent_data:
            print("No agent data available for visualization.")
            return
        
        # Set up the plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LLM Agent Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Agent Success Rates
        agents = list(agent_data.keys())
        success_rates = [agent_data[agent]['success_rate'] * 100 for agent in agents]
        colors = ['green' if rate > 80 else 'orange' if rate > 60 else 'red' for rate in success_rates]
        
        ax1.bar(agents, success_rates, color=colors, alpha=0.7)
        ax1.set_title('Agent Success Rates', fontweight='bold')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(success_rates):
            ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # 2. Response Times
        response_times = [agent_data[agent]['average_response_time'] for agent in agents]
        ax2.bar(agents, response_times, color='skyblue', alpha=0.7)
        ax2.set_title('Average Response Times', fontweight='bold')
        ax2.set_ylabel('Response Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(response_times):
            ax2.text(i, v + 0.01, f'{v:.3f}s', ha='center', va='bottom')
        
        # 3. Quality and Consistency Scores
        quality_scores = [agent_data[agent]['output_quality_score'] for agent in agents]
        consistency_scores = [agent_data[agent]['consistency_score'] for agent in agents]
        
        x = np.arange(len(agents))
        width = 0.35
        
        ax3.bar(x - width/2, quality_scores, width, label='Quality Score', alpha=0.7, color='lightcoral')
        ax3.bar(x + width/2, consistency_scores, width, label='Consistency Score', alpha=0.7, color='lightblue')
        ax3.set_title('Quality vs Consistency Scores', fontweight='bold')
        ax3.set_ylabel('Score (0-1)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(agents, rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # 4. Swarm Performance Radar Chart
        if swarm_data:
            metrics = ['Cohesion', 'Separation', 'Alignment', 'Overall Fitness']
            values = [
                swarm_data.get('cohesion_metric', 0),
                swarm_data.get('separation_metric', 0),
                swarm_data.get('alignment_metric', 0),
                swarm_data.get('overall_fitness', 0)
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2, color='purple')
            ax4.fill(angles, values, alpha=0.25, color='purple')
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics)
            ax4.set_ylim(0, 1)
            ax4.set_title('Swarm Performance Metrics', fontweight='bold')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def export_performance_report(self, filename: str = None):
        """Export detailed performance report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.txt"
        
        if not self.data or "current_report" not in self.data:
            print("No performance data available for export.")
            return
        
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("               LLM SWARM AGENT PERFORMANCE REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            current_report = self.data["current_report"]
            
            # Agent Performance Section
            agent_data = current_report.get("agent_performance", {})
            if agent_data:
                f.write("AGENT PERFORMANCE DETAILS:\n")
                f.write("-" * 30 + "\n")
                for agent_name, metrics in agent_data.items():
                    f.write(f"\n{agent_name}:\n")
                    f.write(f"  Success Rate: {metrics['success_rate']:.2%}\n")
                    f.write(f"  Avg Response Time: {metrics['average_response_time']:.3f}s\n")
                    f.write(f"  Quality Score: {metrics['output_quality_score']:.2f}/1.0\n")
                    f.write(f"  Consistency Score: {metrics['consistency_score']:.2f}/1.0\n")
                    f.write(f"  Total Calls: {metrics['total_calls']}\n")
                    f.write(f"  Failed Calls: {metrics['failed_calls']}\n")
            
            # Swarm Performance Section
            swarm_data = current_report.get("swarm_performance", {})
            if swarm_data:
                f.write(f"\n\nSWARM PERFORMANCE METRICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Cohesion Metric: {swarm_data.get('cohesion_metric', 0):.2f}/1.0\n")
                f.write(f"Separation Metric: {swarm_data.get('separation_metric', 0):.2f}/1.0\n")
                f.write(f"Alignment Metric: {swarm_data.get('alignment_metric', 0):.2f}/1.0\n")
                f.write(f"Overall Fitness: {swarm_data.get('overall_fitness', 0):.2f}/1.0\n")
            
            # Summary Section
            summary = current_report.get("summary", {})
            if summary:
                f.write(f"\n\nSUMMARY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"System Health: {summary.get('overall_system_health', 'UNKNOWN')}\n")
                f.write(f"Average Success Rate: {summary.get('average_success_rate', 0):.2%}\n")
                f.write(f"Average Response Time: {summary.get('average_response_time', 0):.3f}s\n")
                f.write(f"Average Quality Score: {summary.get('average_quality_score', 0):.2f}/1.0\n")
        
        print(f"Performance report exported to: {filename}")

def main():
    """Main function to run performance analysis"""
    print("🔍 LLM Swarm Agent Performance Analyzer")
    print("="*50)
    
    analyzer = PerformanceAnalyzer()
    
    # Run all analysis functions
    analyzer.analyze_agent_performance()
    analyzer.analyze_swarm_behavior()
    analyzer.generate_recommendations()
    
    # Ask user if they want visualizations
    try:
        show_viz = input("\nWould you like to see performance visualizations? (y/n): ").lower().strip()
        if show_viz in ['y', 'yes']:
            analyzer.visualize_performance()
        
        export_report = input("Would you like to export a detailed report? (y/n): ").lower().strip()
        if export_report in ['y', 'yes']:
            analyzer.export_performance_report()
            
    except KeyboardInterrupt:
        print("\nAnalysis completed.")

if __name__ == "__main__":
    main()
