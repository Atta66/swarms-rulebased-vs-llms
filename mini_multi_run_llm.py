#!/usr/bin/env python3
"""
Visual Multi-Run LLM Boids - Full 30 trials with GUI
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from boids_llm import LLMBoids
import os
from datetime import datetime
import time
import psutil
import sys
import argparse
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPUtil not available - GPU monitoring disabled")

# Get current process for accurate monitoring
CURRENT_PROCESS = psutil.Process(os.getpid())

class MiniMultiRunAnalyzer:
    def __init__(self, config_file="config.json", max_trials=None):
        self.config_file = config_file
        self.max_trials = max_trials
        self.load_config()
        self.results = []
        self.timing_data = []
        self.performance_data = []
        
    def load_config(self):
        """Load configuration including random seeds"""
        with open(f"config/{self.config_file}", 'r') as f:
            self.config = json.load(f)
        all_seeds = self.config.get("random_seeds", [12345, 67890, 11111])
        
        # Limit seeds based on max_trials if specified
        if self.max_trials is not None:
            self.seeds = all_seeds[:self.max_trials]
            print(f"📊 Limited to {self.max_trials} trials (out of {len(all_seeds)} available seeds)")
        else:
            self.seeds = all_seeds
            print(f"📊 Using all {len(all_seeds)} available seeds")
    
    def get_system_performance(self):
        """Get current performance metrics for THIS PYTHON PROCESS ONLY"""
        try:
            # Get process-specific CPU and memory usage
            process_cpu = CURRENT_PROCESS.cpu_percent(interval=0.1)
            process_memory = CURRENT_PROCESS.memory_info()
            process_memory_mb = process_memory.rss / (1024 * 1024)  # Convert to MB
            process_memory_percent = CURRENT_PROCESS.memory_percent()
            
            # Get child processes (e.g., if spawning subprocesses)
            children = CURRENT_PROCESS.children(recursive=True)
            total_cpu = process_cpu
            total_memory_mb = process_memory_mb
            
            for child in children:
                try:
                    total_cpu += child.cpu_percent()
                    child_memory = child.memory_info()
                    total_memory_mb += child_memory.rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            perf_data = {
                'process_cpu_percent': total_cpu,
                'process_memory_mb': total_memory_mb,
                'process_memory_percent': process_memory_percent,
                'process_threads': CURRENT_PROCESS.num_threads(),
                'process_pid': CURRENT_PROCESS.pid,
                'num_child_processes': len(children),
                'timestamp': time.time()
            }
            
            # GPU monitoring - this is still system-wide but we'll note it
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        perf_data.update({
                            'gpu_load_percent': gpu.load * 100,
                            'gpu_memory_percent': gpu.memoryUtil * 100,
                            'gpu_memory_used_mb': gpu.memoryUsed,
                            'gpu_temperature': gpu.temperature,
                            'gpu_note': 'system_wide_measurement'
                        })
                    else:
                        perf_data.update({
                            'gpu_load_percent': 0,
                            'gpu_memory_percent': 0,
                            'gpu_memory_used_mb': 0,
                            'gpu_temperature': 0,
                            'gpu_note': 'no_gpu_detected'
                        })
                except Exception as e:
                    print(f"⚠️  GPU monitoring error: {e}")
                    perf_data.update({
                        'gpu_load_percent': 0,
                        'gpu_memory_percent': 0,
                        'gpu_memory_used_mb': 0,
                        'gpu_temperature': 0,
                        'gpu_note': 'monitoring_error'
                    })
            else:
                perf_data.update({
                    'gpu_load_percent': 0,
                    'gpu_memory_percent': 0,
                    'gpu_memory_used_mb': 0,
                    'gpu_temperature': 0,
                    'gpu_note': 'gputil_not_available'
                })
                
        except Exception as e:
            print(f"⚠️  Process monitoring error: {e}")
            perf_data = {
                'process_cpu_percent': 0,
                'process_memory_mb': 0,
                'process_memory_percent': 0,
                'process_threads': 0,
                'process_pid': os.getpid(),
                'num_child_processes': 0,
                'timestamp': time.time(),
                'error': str(e)
            }
        
        return perf_data
        
    def run_single_simulation(self, seed, run_number):
        """Run a single simulation with given seed"""
        print(f"\n🔄 Running LLM simulation {run_number + 1}/{len(self.seeds)} with seed {seed}")
        
        # Start timing and performance monitoring
        start_time = time.time()
        start_perf = self.get_system_performance()
        
        try:
            # Create boids instance with specific seed in headless mode
            boids = LLMBoids(self.config_file, headless=True)
            
            # Run simulation in headless mode
            result_data = boids.run()
            
            # Get performance results
            performance_report = boids.performance_tracker.get_performance_report()
            
            # End timing and performance monitoring
            end_time = time.time()
            end_perf = self.get_system_performance()
            simulation_duration = end_time - start_time
            
            # Store timing data
            timing_info = {
                'seed': seed,
                'run_number': run_number + 1,
                'duration_seconds': simulation_duration,
                'duration_minutes': simulation_duration / 60,
                'start_time': start_time,
                'end_time': end_time
            }
            self.timing_data.append(timing_info)
            
            # Store performance data
            perf_info = {
                'seed': seed,
                'run_number': run_number + 1,
                'start_performance': start_perf,
                'end_performance': end_perf,
                'avg_process_cpu_percent': (start_perf['process_cpu_percent'] + end_perf['process_cpu_percent']) / 2,
                'avg_process_memory_mb': (start_perf['process_memory_mb'] + end_perf['process_memory_mb']) / 2,
                'max_process_memory_mb': max(start_perf['process_memory_mb'], end_perf['process_memory_mb']),
                'avg_process_threads': (start_perf['process_threads'] + end_perf['process_threads']) / 2,
                'avg_gpu_load': (start_perf['gpu_load_percent'] + end_perf['gpu_load_percent']) / 2,
                'avg_gpu_memory': (start_perf['gpu_memory_percent'] + end_perf['gpu_memory_percent']) / 2,
                'max_gpu_temp': max(start_perf['gpu_temperature'], end_perf['gpu_temperature'])
            }
            self.performance_data.append(perf_info)
            
            # Extract key metrics
            result = {
                'seed': seed,
                'run_number': run_number + 1,
                'agent_performance': {},
                'swarm_performance': performance_report.get('swarm_performance', {}),
                'summary': performance_report.get('summary', {})
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
            
            print(f"✅ LLM Simulation {run_number + 1} completed successfully")
            print(f"   Duration: {simulation_duration:.1f}s | Overall Fitness: {result['swarm_performance'].get('overall_fitness', 0):.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ LLM Simulation {run_number + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_simulations(self):
        """Run all simulations with different seeds"""
        print("🚀 Starting Visual Multi-Run Analysis for LLM Boids")
        print(f"📊 Running {len(self.seeds)} simulations with seeds: {self.seeds}")
        print("👁️  Visual mode: You'll see each simulation as it runs!")
        print("⏱️  Timing and performance monitoring enabled")
        
        # Single prompt at the beginning
        if not hasattr(self, 'auto_mode') or not self.auto_mode:
            print(f"\n⏳ Press Enter to start all {len(self.seeds)} simulations...")
            input("   This will run continuously through all trials. Press Enter to begin: ")
            print("🚀 Starting continuous run...")
        else:
            print("\n🤖 Auto mode: Starting all simulations automatically...")
        
        total_start_time = time.time()
        
        for i, seed in enumerate(self.seeds):
            print(f"\n🔄 Starting simulation {i + 1}/{len(self.seeds)} (seed: {seed})")
            
            result = self.run_single_simulation(seed, i)
            if result:
                self.results.append(result)
                progress = ((i + 1) / len(self.seeds)) * 100
                print(f"📈 Progress: {progress:.1f}% ({i + 1}/{len(self.seeds)} completed)")
            else:
                print(f"⚠️  Skipped failed simulation {i + 1}")
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        print(f"\n🎉 Completed {len(self.results)} out of {len(self.seeds)} simulations")
        print(f"⏱️  Total analysis time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        
        # Store total timing
        self.total_analysis_time = total_duration
    
    def print_results(self):
        """Print simple results summary"""
        if not self.results:
            print("❌ No results to display")
            return
            
        print("\n" + "="*60)
        print("           MINI LLM BOIDS ANALYSIS RESULTS")
        print("="*60)
        
        for result in self.results:
            print(f"\nSeed {result['seed']} (Run {result['run_number']}):")
            swarm = result['swarm_performance']
            print(f"  Overall Fitness: {swarm.get('overall_fitness', 0):.3f}")
            print(f"  Cohesion: {swarm.get('cohesion_metric', 0):.3f}")
            print(f"  Separation: {swarm.get('separation_metric', 0):.3f}")
            print(f"  Alignment: {swarm.get('alignment_metric', 0):.3f}")
            
        # Calculate simple averages
        fitness_values = [r['swarm_performance'].get('overall_fitness', 0) for r in self.results]
        cohesion_values = [r['swarm_performance'].get('cohesion_metric', 0) for r in self.results]
        separation_values = [r['swarm_performance'].get('separation_metric', 0) for r in self.results]
        alignment_values = [r['swarm_performance'].get('alignment_metric', 0) for r in self.results]
        
        avg_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        avg_cohesion = np.mean(cohesion_values)
        avg_separation = np.mean(separation_values)
        avg_alignment = np.mean(alignment_values)
        
        print(f"\n🎯 COMPREHENSIVE PERFORMANCE ANALYSIS (n={len(self.results)}):")
        print(f"   Overall Fitness: {avg_fitness:.3f} ± {std_fitness:.3f}")
        print(f"   Cohesion: {avg_cohesion:.3f} ± {np.std(cohesion_values):.3f}")
        print(f"   Separation: {avg_separation:.3f} ± {np.std(separation_values):.3f}")
        print(f"   Alignment: {avg_alignment:.3f} ± {np.std(alignment_values):.3f}")
        print(f"   Consistency: {'High' if std_fitness < 0.1 else 'Moderate' if std_fitness < 0.2 else 'Variable'}")
        
        # Statistical summary
        print(f"\n📊 STATISTICAL SUMMARY:")
        print(f"   Sample Size: n={len(self.results)}")
        print(f"   Min Fitness: {min(fitness_values):.3f}")
        print(f"   Max Fitness: {max(fitness_values):.3f}")
        print(f"   Range: {max(fitness_values) - min(fitness_values):.3f}")
        
        # COMPREHENSIVE TIMING AND PERFORMANCE ANALYSIS
        if self.timing_data:
            durations = [t['duration_seconds'] for t in self.timing_data]
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            total_time = getattr(self, 'total_analysis_time', 0)
            
            print(f"\n⏱️  COMPREHENSIVE TIMING ANALYSIS:")
            print(f"   Average per simulation: {avg_duration:.2f} ± {std_duration:.2f} seconds")
            print(f"   Average per simulation: {avg_duration/60:.2f} ± {std_duration/60:.2f} minutes")
            print(f"   Fastest simulation: {min(durations):.2f} seconds ({min(durations)/60:.2f} minutes)")
            print(f"   Slowest simulation: {max(durations):.2f} seconds ({max(durations)/60:.2f} minutes)")
            print(f"   Total analysis time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"   Time efficiency: {(sum(durations)/total_time)*100:.1f}% (simulation vs total time)")
        
        # COMPREHENSIVE PERFORMANCE ANALYSIS
        if self.performance_data:
            cpu_values = [p['avg_process_cpu_percent'] for p in self.performance_data]
            memory_mb_values = [p['avg_process_memory_mb'] for p in self.performance_data]
            max_memory_mb_values = [p['max_process_memory_mb'] for p in self.performance_data]
            thread_values = [p['avg_process_threads'] for p in self.performance_data]
            gpu_load_values = [p['avg_gpu_load'] for p in self.performance_data]
            gpu_memory_values = [p['avg_gpu_memory'] for p in self.performance_data]
            gpu_temp_values = [p['max_gpu_temp'] for p in self.performance_data if p['max_gpu_temp'] > 0]
            
            avg_process_cpu = np.mean(cpu_values)
            std_process_cpu = np.std(cpu_values)
            avg_process_memory_mb = np.mean(memory_mb_values)
            std_process_memory_mb = np.std(memory_mb_values)
            max_process_memory_mb = max(max_memory_mb_values)
            avg_threads = np.mean(thread_values)
            avg_gpu_load = np.mean(gpu_load_values)
            std_gpu_load = np.std(gpu_load_values)
            avg_gpu_memory = np.mean(gpu_memory_values)
            std_gpu_memory = np.std(gpu_memory_values)
            
            print(f"\n🖥️  PROCESS-SPECIFIC PERFORMANCE (PID: {CURRENT_PROCESS.pid}):")
            print(f"   Python Process CPU: {avg_process_cpu:.1f}% ± {std_process_cpu:.1f}% (range: {min(cpu_values):.1f}%-{max(cpu_values):.1f}%)")
            print(f"   Python Process Memory: {avg_process_memory_mb:.1f} ± {std_process_memory_mb:.1f} MB")
            print(f"   Peak Memory Usage: {max_process_memory_mb:.1f} MB")
            print(f"   Average Threads: {avg_threads:.1f}")
            
            if GPU_AVAILABLE and avg_gpu_load > 0:
                print(f"   System GPU Load: {avg_gpu_load:.1f}% ± {std_gpu_load:.1f}% (range: {min(gpu_load_values):.1f}%-{max(gpu_load_values):.1f}%)")
                print(f"   System GPU Memory: {avg_gpu_memory:.1f}% ± {std_gpu_memory:.1f}% (range: {min(gpu_memory_values):.1f}%-{max(gpu_memory_values):.1f}%)")
                print(f"   Note: GPU metrics are system-wide, not process-specific")
                
                if gpu_temp_values:
                    avg_gpu_temp = np.mean(gpu_temp_values)
                    max_gpu_temp = max(gpu_temp_values)
                    print(f"   GPU Temperature: {avg_gpu_temp:.1f}°C average, {max_gpu_temp:.1f}°C maximum")
            else:
                print(f"   GPU: Not utilized or not available")
            
            # Performance efficiency analysis
            print(f"\n📈 PROCESS PERFORMANCE ANALYSIS:")
            if std_process_cpu < 5:
                cpu_stability = "Very Stable"
            elif std_process_cpu < 15:
                cpu_stability = "Stable"
            elif std_process_cpu < 30:
                cpu_stability = "Moderate variation"
            else:
                cpu_stability = "High variation"
            
            print(f"   CPU Usage Pattern: {cpu_stability} (std: {std_process_cpu:.1f}%)")
            print(f"   Memory Efficiency: {avg_process_memory_mb:.0f} MB average, {max_process_memory_mb:.0f} MB peak")
            print(f"   Memory Growth: {max_process_memory_mb - min(memory_mb_values):.1f} MB during execution")
            
            if avg_process_cpu > 50:
                cpu_assessment = "High CPU utilization - computationally intensive"
            elif avg_process_cpu > 20:
                cpu_assessment = "Moderate CPU utilization - balanced processing"
            else:
                cpu_assessment = "Low CPU utilization - I/O or waiting dominant"
            
            print(f"   CPU Assessment: {cpu_assessment}")
            
            if GPU_AVAILABLE and avg_gpu_load > 5:
                print(f"   GPU Utilization: {'Good' if avg_gpu_load > 50 else 'Low'} ({avg_gpu_load:.1f}% average)")
                print(f"   Note: This is system-wide GPU usage, may include other applications")
            else:
                print(f"   GPU Utilization: Minimal or CPU-only processing")
                
            # Process-specific insights
            print(f"\n🔍 PROCESS INSIGHTS:")
            print(f"   Process ID: {CURRENT_PROCESS.pid}")
            print(f"   Thread Usage: {avg_threads:.1f} threads on average")
            if max_process_memory_mb > 1000:
                print(f"   Memory Profile: High memory usage ({max_process_memory_mb:.0f} MB peak)")
            elif max_process_memory_mb > 500:
                print(f"   Memory Profile: Moderate memory usage ({max_process_memory_mb:.0f} MB peak)")
            else:
                print(f"   Memory Profile: Low memory usage ({max_process_memory_mb:.0f} MB peak)")
                
            print(f"   Resource Attribution: All CPU/Memory metrics are specific to this Python process")
        
    def save_results(self):
        """Save results to JSON file"""
        os.makedirs("results/visual_llm_boids", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/visual_llm_boids/visual_multi_run_{timestamp}.json"
        
        output_data = {
            'metadata': {
                'timestamp': timestamp,
                'config_file': self.config_file,
                'seeds_used': [r['seed'] for r in self.results],
                'successful_runs': len(self.results),
                'total_attempted': len(self.seeds),
                'mode': 'visual_gui',
                'total_analysis_time_seconds': getattr(self, 'total_analysis_time', 0),
                'gpu_available': GPU_AVAILABLE
            },
            'raw_results': self.results,
            'timing_data': self.timing_data,
            'performance_data': self.performance_data
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename

def get_trial_count():
    """Get the number of trials to run from user input"""
    try:
        # Load config to see available seeds
        with open("config/config.json", 'r') as f:
            config = json.load(f)
        max_available = len(config.get("random_seeds", []))
        
        print(f"\n🎯 TRIAL SELECTION:")
        print(f"   Available seeds: {max_available}")
        print(f"   Recommended options:")
        print(f"   • 3 trials  - Quick test")
        print(f"   • 10 trials - Medium analysis")
        print(f"   • 20 trials - Good statistical power")
        print(f"   • 30 trials - Full analysis (maximum)")
        
        while True:
            try:
                choice = input(f"\nHow many trials do you want to run? (1-{max_available}): ").strip()
                
                if choice.lower() in ['q', 'quit', 'exit']:
                    print("Exiting...")
                    return None
                
                num_trials = int(choice)
                
                if 1 <= num_trials <= max_available:
                    estimated_time_min = num_trials * 1.5
                    estimated_time_max = num_trials * 3
                    
                    print(f"\n✅ Selected: {num_trials} trials")
                    print(f"⏱️  Estimated time: ~{estimated_time_min:.0f}-{estimated_time_max:.0f} minutes")
                    
                    confirm = input("Continue? (y/n): ").lower().strip()
                    if confirm in ['y', 'yes']:
                        return num_trials
                    elif confirm in ['n', 'no']:
                        continue
                    else:
                        print("Please enter 'y' or 'n'")
                        continue
                else:
                    print(f"❌ Please enter a number between 1 and {max_available}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
                
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return None

def main():
    """Main execution function"""
    print("🔍 Visual LLM Boids Multi-Run Performance Analyzer")
    print("👁️  This will run simulations with GUI visible!")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM Boids Multi-Run Analyzer')
    parser.add_argument('-n', '--trials', type=int, help='Number of trials to run (1-30)')
    parser.add_argument('--auto', action='store_true', help='Run without interactive prompts')
    args = parser.parse_args()
    
    try:
        # Determine number of trials
        if args.trials:
            # Use command line argument
            num_trials = args.trials
            with open("config/config.json", 'r') as f:
                config = json.load(f)
            max_available = len(config.get("random_seeds", []))
            
            if num_trials < 1 or num_trials > max_available:
                print(f"❌ Number of trials must be between 1 and {max_available}")
                return
                
            print(f"📊 Command line: Running {num_trials} trials")
            
        else:
            # Interactive selection
            num_trials = get_trial_count()
            if num_trials is None:
                return
        
        # Create analyzer with specified number of trials
        analyzer = MiniMultiRunAnalyzer(max_trials=num_trials)
        print(f"✅ Analyzer initialized with {len(analyzer.seeds)} seeds")
        
        # Run simulations
        if args.auto:
            print("🤖 Auto mode: Running without interactive prompts")
            # Modify analyzer to skip input prompts in auto mode
            analyzer.auto_mode = True
        
        analyzer.run_all_simulations()
        
        if analyzer.results:
            analyzer.print_results()
            analyzer.save_results()
            print("\n✅ Visual analysis completed successfully!")
        else:
            print("❌ No results were generated. Check for errors above.")
            
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
