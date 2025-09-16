#!/usr/bin/env python3
"""
Multi-Run Performance Analysis for Classic Boids

This script runs multiple simulations with different random seeds
to get statistical performance data for classic boids.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from boids import ClassicBoids
import os
import time
import psutil
import threading
from datetime import datetime

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class MultiRunAnalyzer:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.load_config()
        self.results = []
        
    def load_config(self):
        """Load configuration including random seeds"""
        with open(f"config/{self.config_file}", 'r') as f:
            self.config = json.load(f)
        self.seeds = self.config.get("random_seeds", [12345, 67890, 11111, 22222, 33333])
        
    def get_system_performance(self):
        """Get current system performance metrics"""
        try:
            # Get process-specific CPU and memory for this Python process
            current_process = psutil.Process()
            
            performance_data = {
                'cpu_percent': current_process.cpu_percent(),
                'memory_mb': current_process.memory_info().rss / (1024 * 1024),
                'threads': current_process.num_threads(),
                'timestamp': time.time()
            }
            
            # Add GPU metrics if available
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        performance_data['gpu_load'] = gpu.load * 100
                        performance_data['gpu_memory'] = gpu.memoryUsed / gpu.memoryTotal * 100
                        performance_data['gpu_temp'] = gpu.temperature
                    else:
                        performance_data['gpu_load'] = 0
                        performance_data['gpu_memory'] = 0
                        performance_data['gpu_temp'] = 0
                except Exception as e:
                    performance_data['gpu_load'] = 0
                    performance_data['gpu_memory'] = 0
                    performance_data['gpu_temp'] = 0
            else:
                performance_data['gpu_load'] = 0
                performance_data['gpu_memory'] = 0
                performance_data['gpu_temp'] = 0
                
            return performance_data
            
        except Exception as e:
            print(f"⚠️  Warning: Could not get performance data: {e}")
            return {
                'cpu_percent': 0,
                'memory_mb': 0,
                'threads': 0,
                'gpu_load': 0,
                'gpu_memory': 0,
                'gpu_temp': 0,
                'timestamp': time.time()
            }
        
    def run_single_simulation(self, seed, run_number):
        """Run a single simulation with given seed, including timing and performance monitoring"""
        print(f"\n🔄 Running simulation {run_number + 1}/{len(self.seeds)} with seed {seed}")
        
        # Start timing and performance monitoring
        start_time = time.time()
        start_perf = self.get_system_performance()
        
        try:
            # Create boids instance with specific seed in headless mode for efficiency
            boids = ClassicBoids(self.config_file, seed=seed, headless=True)
            
            # Run simulation without GUI (headless)
            boids.run_headless()
            
            # Get performance results
            performance_report = boids.performance_tracker.get_performance_report()
            
            # End timing and performance monitoring
            end_time = time.time()
            end_perf = self.get_system_performance()
            simulation_duration = end_time - start_time
            
            # Extract key metrics
            result = {
                'seed': seed,
                'run_number': run_number + 1,
                'agent_performance': {},
                'swarm_performance': performance_report.get('swarm_performance', {}),
                'summary': performance_report.get('summary', {}),
                'timing': {
                    'duration_seconds': simulation_duration,
                    'duration_minutes': simulation_duration / 60,
                    'start_time': start_time,
                    'end_time': end_time
                },
                'performance': {
                    'start': start_perf,
                    'end': end_perf,
                    'cpu_change': end_perf['cpu_percent'] - start_perf['cpu_percent'],
                    'memory_change_mb': end_perf['memory_mb'] - start_perf['memory_mb'],
                    'avg_cpu': (start_perf['cpu_percent'] + end_perf['cpu_percent']) / 2,
                    'avg_memory_mb': (start_perf['memory_mb'] + end_perf['memory_mb']) / 2,
                    'avg_threads': (start_perf['threads'] + end_perf['threads']) / 2,
                    'avg_gpu_load': (start_perf['gpu_load'] + end_perf['gpu_load']) / 2,
                    'avg_gpu_memory': (start_perf['gpu_memory'] + end_perf['gpu_memory']) / 2,
                    'max_gpu_temp': max(start_perf['gpu_temp'], end_perf['gpu_temp'])
                }
            }
            
            # Extract agent performance
            for agent_name, metrics in performance_report.get('agent_performance', {}).items():
                result['agent_performance'][agent_name] = {
                    'success_rate': metrics.get('success_rate', 0),
                    'avg_response_time': metrics.get('average_response_time', 0),
                    'output_quality_score': metrics.get('output_quality_score', 0),
                    'consistency_score': metrics.get('consistency_score', 0),
                    'total_calls': metrics.get('total_calls', 0)
                }
            
            # Print timing info
            overall_fitness = result['swarm_performance'].get('overall_fitness', 0)
            print(f"✅ Classic Simulation {run_number + 1} completed successfully")
            print(f"   Duration: {simulation_duration:.1f}s | Overall Fitness: {overall_fitness:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Simulation {run_number + 1} failed: {e}")
            return None
    
    def run_all_simulations(self):
        """Run all simulations with different seeds, including comprehensive timing analysis"""
        print("🚀 Starting Multi-Run Analysis for Classic Boids")
        print(f"📊 Running {len(self.seeds)} simulations with seeds: {self.seeds}")
        print("⏱️  Timing and performance monitoring enabled")
        
        # Start overall timing
        overall_start_time = time.time()
        successful_runs = 0
        
        for i, seed in enumerate(self.seeds):
            try:
                result = self.run_single_simulation(seed, i)
                if result:
                    self.results.append(result)
                    successful_runs += 1
                    
                    progress = (successful_runs / len(self.seeds)) * 100
                    print(f"📈 Progress: {progress:.1f}% ({successful_runs}/{len(self.seeds)} completed)")
                    
                    # Print quick summary
                    swarm = result['swarm_performance']
                    print(f"   Cohesion: {swarm.get('cohesion_metric', 0):.3f}, "
                          f"Separation: {swarm.get('separation_metric', 0):.3f}, "
                          f"Alignment: {swarm.get('alignment_metric', 0):.3f}")
                      
            except Exception as e:
                print(f"❌ Simulation {i + 1} failed: {e}")
                continue
        
        # Calculate overall timing
        overall_end_time = time.time()
        total_analysis_time = overall_end_time - overall_start_time
        
        print(f"\n🎉 Completed {successful_runs} out of {len(self.seeds)} simulations")
        print(f"⏱️  Total analysis time: {total_analysis_time:.2f} seconds ({total_analysis_time/60:.2f} minutes)")
        
        if successful_runs > 0:
            # Calculate timing statistics
            durations = [r['timing']['duration_seconds'] for r in self.results if r]
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            min_duration = np.min(durations)
            max_duration = np.max(durations)
            
            print(f"\n⏱️  COMPREHENSIVE TIMING ANALYSIS:")
            print(f"   Average per simulation: {avg_duration:.2f} ± {std_duration:.2f} seconds")
            print(f"   Average per simulation: {avg_duration/60:.2f} ± {std_duration/60:.2f} minutes")
            print(f"   Fastest simulation: {min_duration:.2f} seconds ({min_duration/60:.2f} minutes)")
            print(f"   Slowest simulation: {max_duration:.2f} seconds ({max_duration/60:.2f} minutes)")
            print(f"   Time efficiency: {(sum(durations)/total_analysis_time)*100:.1f}% (simulation vs total time)")
            
            # Calculate performance statistics
            cpu_values = [r['performance']['avg_cpu'] for r in self.results if r]
            memory_values = [r['performance']['avg_memory_mb'] for r in self.results if r]
            gpu_load_values = [r['performance']['avg_gpu_load'] for r in self.results if r]
            gpu_memory_values = [r['performance']['avg_gpu_memory'] for r in self.results if r]
            gpu_temp_values = [r['performance']['max_gpu_temp'] for r in self.results if r]
            threads_values = [r['performance']['avg_threads'] for r in self.results if r]
            
            if cpu_values:
                print(f"\n🖥️  PROCESS-SPECIFIC PERFORMANCE:")
                current_process = psutil.Process()
                print(f"   Python Process CPU: {np.mean(cpu_values):.1f}% ± {np.std(cpu_values):.1f}% (range: {np.min(cpu_values):.1f}%-{np.max(cpu_values):.1f}%)")
                print(f"   Python Process Memory: {np.mean(memory_values):.1f} ± {np.std(memory_values):.1f} MB")
                print(f"   Peak Memory Usage: {np.max(memory_values):.1f} MB")
                print(f"   Average Threads: {np.mean(threads_values):.1f}")
                
                if GPU_AVAILABLE and any(gpu_load_values):
                    print(f"   System GPU Load: {np.mean(gpu_load_values):.1f}% ± {np.std(gpu_load_values):.1f}% (range: {np.min(gpu_load_values):.1f}%-{np.max(gpu_load_values):.1f}%)")
                    print(f"   System GPU Memory: {np.mean(gpu_memory_values):.1f}% ± {np.std(gpu_memory_values):.1f}% (range: {np.min(gpu_memory_values):.1f}%-{np.max(gpu_memory_values):.1f}%)")
                    print(f"   Note: GPU metrics are system-wide, not process-specific")
                    if any(gpu_temp_values):
                        print(f"   GPU Temperature: {np.mean(gpu_temp_values):.1f}°C average, {np.max(gpu_temp_values):.1f}°C maximum")
                
                # Performance analysis
                cpu_std = np.std(cpu_values)
                memory_growth = np.max(memory_values) - np.min(memory_values)
                print(f"\n📈 PROCESS PERFORMANCE ANALYSIS:")
                print(f"   CPU Usage Pattern: {'Very Stable' if cpu_std < 1 else 'Stable' if cpu_std < 3 else 'Variable'} (std: {cpu_std:.1f}%)")
                print(f"   Memory Efficiency: {np.mean(memory_values):.0f} MB average, {np.max(memory_values):.0f} MB peak")
                print(f"   Memory Growth: {memory_growth:.1f} MB during execution")
                
                avg_cpu = np.mean(cpu_values)
                if avg_cpu < 5:
                    print(f"   CPU Assessment: Low CPU utilization - I/O or waiting dominant")
                elif avg_cpu < 25:
                    print(f"   CPU Assessment: Moderate CPU usage - well balanced")
                else:
                    print(f"   CPU Assessment: High CPU usage - computation intensive")
                    
                if GPU_AVAILABLE and any(gpu_load_values):
                    avg_gpu = np.mean(gpu_load_values)
                    print(f"   GPU Utilization: {'Low' if avg_gpu < 30 else 'Moderate' if avg_gpu < 70 else 'High'} ({avg_gpu:.1f}% average)")
                    print(f"   Note: This is system-wide GPU usage, may include other applications")
                
                print(f"\n🔍 PROCESS INSIGHTS:")
                print(f"   Process ID: {current_process.pid}")
                print(f"   Thread Usage: {np.mean(threads_values):.1f} threads on average")
                print(f"   Memory Profile: {'Low' if np.max(memory_values) < 500 else 'Moderate' if np.max(memory_values) < 1000 else 'High'} memory usage ({np.max(memory_values):.0f} MB peak)")
                print(f"   Resource Attribution: All CPU/Memory metrics are specific to this Python process")
        
        return successful_runs > 0
        
    def calculate_statistics(self):
        """Calculate mean, std, min, max for all metrics"""
        if not self.results:
            return {}
            
        stats = {
            'agent_stats': {},
            'swarm_stats': {},
            'summary_stats': {}
        }
        
        # Agent performance statistics
        agent_names = ['ClassicSeparation', 'ClassicCohesion', 'ClassicAlignment']
        
        for agent_name in agent_names:
            if agent_name in self.results[0]['agent_performance']:
                agent_data = {}
                for metric in ['success_rate', 'avg_response_time', 'output_quality_score', 'consistency_score']:
                    values = [r['agent_performance'][agent_name][metric] for r in self.results]
                    agent_data[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                stats['agent_stats'][agent_name] = agent_data
        
        # Swarm performance statistics
        for metric in ['cohesion_metric', 'separation_metric', 'alignment_metric', 'overall_fitness']:
            values = [r['swarm_performance'].get(metric, 0) for r in self.results]
            stats['swarm_stats'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return stats
    
    def print_results(self):
        """Print comprehensive results"""
        stats = self.calculate_statistics()
        
        print("\n" + "="*80)
        print("                  MULTI-RUN CLASSIC BOIDS ANALYSIS")
        print("="*80)
        print(f"Simulations completed: {len(self.results)}")
        print(f"Seeds used: {[r['seed'] for r in self.results]}")
        
        # Agent Performance Statistics
        print(f"\n📊 AGENT PERFORMANCE STATISTICS (n={len(self.results)}):")
        print("-" * 70)
        
        for agent_name, agent_stats in stats['agent_stats'].items():
            print(f"\n🤖 {agent_name}:")
            for metric, values in agent_stats.items():
                metric_name = metric.replace('_', ' ').title()
                if 'rate' in metric or 'score' in metric:
                    print(f"   {metric_name:<20}: {values['mean']:.3f} ± {values['std']:.3f} "
                          f"(min: {values['min']:.3f}, max: {values['max']:.3f})")
                else:
                    print(f"   {metric_name:<20}: {values['mean']:.4f} ± {values['std']:.4f} "
                          f"(min: {values['min']:.4f}, max: {values['max']:.4f})")
        
        # Swarm Performance Statistics  
        print(f"\n🌐 SWARM PERFORMANCE STATISTICS (n={len(self.results)}):")
        print("-" * 70)
        
        for metric, values in stats['swarm_stats'].items():
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<20}: {values['mean']:.3f} ± {values['std']:.3f} "
                  f"(min: {values['min']:.3f}, max: {values['max']:.3f})")
        
        # Performance Assessment
        print(f"\n💡 PERFORMANCE ASSESSMENT:")
        print("-" * 40)
        
        swarm_stats = stats['swarm_stats']
        overall_mean = swarm_stats['overall_fitness']['mean']
        overall_std = swarm_stats['overall_fitness']['std']
        
        print(f"Overall Fitness: {overall_mean:.3f} ± {overall_std:.3f}")
        
        if overall_mean > 0.8:
            assessment = "EXCELLENT - Consistently high performance"
        elif overall_mean > 0.6:
            assessment = "GOOD - Reliable performance with room for improvement"
        elif overall_mean > 0.4:
            assessment = "MODERATE - Variable performance, needs optimization"
        else:
            assessment = "POOR - Significant issues detected"
            
        print(f"Assessment: {assessment}")
        
        if overall_std > 0.2:
            print("⚠️  High variability detected - consider parameter tuning")
        elif overall_std < 0.05:
            print("✅ Low variability - consistent performance across runs")
        else:
            print("✅ Moderate variability - acceptable performance range")
    
    def save_results(self):
        """Save results to JSON file in results/classic_boids folder"""
        # Ensure results directory exists
        results_dir = "results/classic_boids"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/multi_run_classic_boids_{timestamp}.json"
        
        # Calculate timing and performance statistics
        timing_data = {}
        performance_data = {}
        
        if self.results:
            durations = [r['timing']['duration_seconds'] for r in self.results if r and 'timing' in r]
            cpu_values = [r['performance']['avg_cpu'] for r in self.results if r and 'performance' in r]
            memory_values = [r['performance']['avg_memory_mb'] for r in self.results if r and 'performance' in r]
            
            if durations:
                timing_data = {
                    'total_simulations': len(durations),
                    'average_duration_seconds': np.mean(durations),
                    'std_duration_seconds': np.std(durations),
                    'min_duration_seconds': np.min(durations),
                    'max_duration_seconds': np.max(durations),
                    'total_simulation_time': np.sum(durations)
                }
            
            if cpu_values:
                performance_data = {
                    'process_cpu': {
                        'mean': np.mean(cpu_values),
                        'std': np.std(cpu_values),
                        'min': np.min(cpu_values),
                        'max': np.max(cpu_values)
                    },
                    'process_memory_mb': {
                        'mean': np.mean(memory_values),
                        'std': np.std(memory_values),
                        'min': np.min(memory_values),
                        'max': np.max(memory_values),
                        'peak': np.max(memory_values)
                    },
                    'gpu_available': GPU_AVAILABLE
                }
                
                # Add GPU stats if available
                if GPU_AVAILABLE:
                    gpu_load_values = [r['performance']['avg_gpu_load'] for r in self.results if r and 'performance' in r]
                    gpu_memory_values = [r['performance']['avg_gpu_memory'] for r in self.results if r and 'performance' in r]
                    if gpu_load_values:
                        performance_data['system_gpu'] = {
                            'load_percent': {
                                'mean': np.mean(gpu_load_values),
                                'std': np.std(gpu_load_values),
                                'min': np.min(gpu_load_values),
                                'max': np.max(gpu_load_values)
                            },
                            'memory_percent': {
                                'mean': np.mean(gpu_memory_values),
                                'std': np.std(gpu_memory_values),
                                'min': np.min(gpu_memory_values),
                                'max': np.max(gpu_memory_values)
                            }
                        }
        
        output_data = {
            'metadata': {
                'timestamp': timestamp,
                'config_file': self.config_file,
                'seeds_used': [r['seed'] for r in self.results],
                'successful_runs': len(self.results),
                'total_attempted': len(self.seeds),
                'analysis_type': 'Classic_Boids_Multi_Run',
                'timing_enabled': True,
                'performance_monitoring': True,
                'gpu_available': GPU_AVAILABLE
            },
            'statistics': self.calculate_statistics(),
            'timing_data': timing_data,
            'performance_data': performance_data,
            'raw_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename
    
    def visualize_results(self):
        """Create visualization of results"""
        if len(self.results) < 2:
            print("❌ Need at least 2 results for visualization")
            return
            
        stats = self.calculate_statistics()
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Classic Boids Multi-Run Analysis (n={len(self.results)})', fontsize=16)
        
        # 1. Swarm Performance Means
        swarm_metrics = ['cohesion_metric', 'separation_metric', 'alignment_metric', 'overall_fitness']
        swarm_labels = ['Cohesion', 'Separation', 'Alignment', 'Overall Fitness']
        swarm_means = [stats['swarm_stats'][metric]['mean'] for metric in swarm_metrics]
        swarm_stds = [stats['swarm_stats'][metric]['std'] for metric in swarm_metrics]
        
        bars1 = ax1.bar(swarm_labels, swarm_means, yerr=swarm_stds, capsize=5, alpha=0.7, color='skyblue')
        ax1.set_title('Swarm Performance Metrics')
        ax1.set_ylabel('Score (0-1)')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, mean in zip(bars1, swarm_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 2. Agent Consistency Comparison
        agent_names = ['ClassicSeparation', 'ClassicCohesion', 'ClassicAlignment']
        agent_labels = ['Separation', 'Cohesion', 'Alignment']
        
        if agent_names[0] in stats['agent_stats']:
            consistency_means = [stats['agent_stats'][agent]['consistency_score']['mean'] 
                               for agent in agent_names]
            consistency_stds = [stats['agent_stats'][agent]['consistency_score']['std'] 
                              for agent in agent_names]
            
            bars2 = ax2.bar(agent_labels, consistency_means, yerr=consistency_stds, 
                           capsize=5, alpha=0.7, color='lightcoral')
            ax2.set_title('Agent Consistency Scores')
            ax2.set_ylabel('Consistency (0-1)')
            ax2.set_ylim(0, 1.1)
            
            for bar, mean in zip(bars2, consistency_means):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{mean:.3f}', ha='center', va='bottom')
        
        # 3. Performance Variability
        variability_metrics = swarm_metrics
        variability_values = [stats['swarm_stats'][metric]['std'] for metric in variability_metrics]
        
        bars3 = ax3.bar(swarm_labels, variability_values, alpha=0.7, color='gold')
        ax3.set_title('Performance Variability (Standard Deviation)')
        ax3.set_ylabel('Standard Deviation')
        
        for bar, std in zip(bars3, variability_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{std:.3f}', ha='center', va='bottom')
        
        # 4. Run-by-Run Overall Fitness
        run_numbers = [r['run_number'] for r in self.results]
        fitness_values = [r['swarm_performance'].get('overall_fitness', 0) for r in self.results]
        
        ax4.plot(run_numbers, fitness_values, 'o-', linewidth=2, markersize=8, color='green')
        ax4.set_title('Overall Fitness by Run')
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
        viz_filename = f'{viz_dir}/classic_boids_multi_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {viz_filename}")
        plt.show()

def main():
    """Main execution function"""
    print("🔍 Classic Boids Multi-Run Performance Analyzer")
    
    analyzer = MultiRunAnalyzer()
    analyzer.run_all_simulations()
    analyzer.print_results()
    analyzer.save_results()
    
    try:
        show_viz = input("\nShow visualization? (y/n): ").lower().strip()
        if show_viz in ['y', 'yes']:
            analyzer.visualize_results()
    except KeyboardInterrupt:
        print("\nAnalysis completed.")

if __name__ == "__main__":
    main()
