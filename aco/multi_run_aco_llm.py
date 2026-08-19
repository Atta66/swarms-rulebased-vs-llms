#!/usr/bin/env python3
"""
Multi-Run ACO LLM Analysis - 30 trials with comprehensive statistics
Parallel to regular ACO multi-run analysis for direct comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os
from datetime import datetime
import random
from aco_llm_enhanced import ACOLLMEnhanced

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPUtil not available - GPU monitoring disabled")

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class MultiRunACOLLMAnalyzer:
    def __init__(self, config_file="config/aco_config.json"):
        self.config_file = config_file
        self.load_config()
        self.results = []
        self.timing_data = []
        self.performance_data = []
        self.total_start_time = None
        self.total_end_time = None
        
        # Get current process for monitoring
        self.current_process = psutil.Process()
        print(f"🔍 Monitoring process PID: {self.current_process.pid}")
        
    def load_config(self):
        """Load configuration including random seeds"""
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        
        # Get number of trials from config (matching regular ACO)
        num_trials = self.config.get("num_trials", 30)
        
        # Generate seeds if not enough provided
        predefined_seeds = self.config.get("random_seeds", [12345, 67890, 11111, 22222, 33333])
        
        if len(predefined_seeds) >= num_trials:
            self.seeds = predefined_seeds[:num_trials]
        else:
            # Use predefined seeds and generate additional ones
            self.seeds = predefined_seeds.copy()
            np.random.seed(42)  # For reproducible seed generation
            while len(self.seeds) < num_trials:
                new_seed = np.random.randint(10000, 99999)
                if new_seed not in self.seeds:
                    self.seeds.append(new_seed)
        
        print(f"📊 Loaded config: {len(self.seeds)} trials, {self.config['max_iterations']} iterations per trial")
        if len(self.seeds) > 5:
            print(f"🎲 Seeds: {self.seeds[:5]} ... (showing first 5 of {len(self.seeds)})")
        else:
            print(f"🎲 Seeds: {self.seeds}")
        
    def get_process_performance(self):
        """Get current process-specific performance metrics"""
        try:
            # Process-specific metrics
            cpu_percent = self.current_process.cpu_percent(interval=0.1)
            memory_info = self.current_process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self.current_process.memory_percent()
            threads = self.current_process.num_threads()
            
            # GPU monitoring (system-wide, but process-aware)
            gpu_load = 0
            gpu_memory = 0
            gpu_temp = 0
            gpu_memory_used_mb = 0
            
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_load = gpu.load * 100
                        gpu_memory = gpu.memoryUtil * 100
                        gpu_temp = gpu.temperature
                        gpu_memory_used_mb = gpu.memoryUsed
                except Exception as e:
                    print(f"⚠️  GPU monitoring error: {e}")
            
            return {
                'process_cpu_percent': cpu_percent,
                'process_memory_mb': memory_mb,
                'process_memory_percent': memory_percent,
                'process_threads': threads,
                'process_pid': self.current_process.pid,
                'num_child_processes': len(self.current_process.children()),
                'timestamp': time.time(),
                'gpu_load_percent': gpu_load,
                'gpu_memory_percent': gpu_memory,
                'gpu_memory_used_mb': gpu_memory_used_mb,
                'gpu_temperature': gpu_temp,
                'gpu_note': 'system_wide_measurement'
            }
        except Exception as e:
            print(f"⚠️  Performance monitoring error: {e}")
            return {
                'process_cpu_percent': 0, 'process_memory_mb': 0, 'process_memory_percent': 0,
                'process_threads': 1, 'process_pid': 0, 'num_child_processes': 0,
                'timestamp': time.time(), 'gpu_load_percent': 0, 'gpu_memory_percent': 0,
                'gpu_memory_used_mb': 0, 'gpu_temperature': 0, 'gpu_note': 'unavailable'
            }
    
    def run_single_aco_llm(self, seed, run_number):
        """Run a single ACO LLM simulation with given seed"""
        print(f"🤖 Running ACO LLM simulation {run_number}/{len(self.seeds)} with seed {seed}")
        
        # Start timing and performance monitoring
        start_time = time.time()
        start_perf = self.get_process_performance()
        
        try:
            # Initialize ACO LLM
            aco_llm = ACOLLMEnhanced(config_file="config/config.json")
            
            # Configuration values
            max_iterations = self.config["max_iterations"]
            target_ratio = self.config["target_ratio"]
            ratio_tolerance = self.config["ratio_tolerance"]
            
            # Get additional parameters for detailed tracking
            distance_weight = self.config.get("distance_weight", 1.0)
            pheromone_deposit = self.config.get("pheromone_deposit", 0.5)
            
            # Run simulation
            final_step = aco_llm.simulate(
                max_iterations=max_iterations,
                target_ratio=target_ratio,
                ratio_tolerance=ratio_tolerance,
                seed=seed,
                show_visualization=False
            )
            
            # Get results
            simulation_results = aco_llm.get_results()
            
        except Exception as e:
            print(f"❌ Error in ACO LLM simulation: {e}")
            raise
        
        # End timing and performance monitoring
        end_time = time.time()
        end_perf = self.get_process_performance()
        simulation_duration = end_time - start_time
        
        # Extract data for metrics calculation
        paths = simulation_results['final_pheromone_levels']
        path_selection_history = simulation_results['path_selection_history']
        step_wise_ratios = simulation_results['step_wise_ratios']
        pheromone_history = simulation_results['pheromone_history']
        
        # Create detailed step data for visualizer compatibility
        detailed_step_data = self.create_detailed_step_data_from_llm_results(
            pheromone_history, 
            path_selection_history, 
            paths,
            distance_weight,
            final_step
        )
        
        # Calculate phase-based selection analysis (matching visualizer approach)
        phase_analysis = self.calculate_phase_selections(detailed_step_data, max_iterations)
        
        # Calculate performance metrics using same methods as regular ACO
        convergence_speed = self.calculate_convergence_speed(
            path_selection_history["short"], 
            path_selection_history["long"],
            max_iterations
        )
        solution_quality = self.calculate_solution_quality(paths)
        learning_efficiency = self.calculate_learning_progression(
            path_selection_history["short"], 
            path_selection_history["long"],
            max_iterations
        )
        learning_stability = self.calculate_learning_stability(step_wise_ratios)
        
        # Overall fitness: 50% convergence speed + 50% learning progression
        overall_fitness = (
            convergence_speed * 0.5 +      # 50% - How quickly it converges
            learning_efficiency * 0.5      # 50% - Exploration→Transition→Exploitation progression
        )
        
        # Store results
        result = {
            'seed': seed,
            'run_number': run_number,
            'algorithm': 'ACO_LLM',  # Identify this as LLM version for visualizer
            'final_step': final_step,
            'convergence_time_seconds': simulation_duration,
            'swarm_performance': {
                'convergence_speed': convergence_speed,
                'solution_quality': solution_quality,
                'learning_efficiency': learning_efficiency,
                'learning_stability': learning_stability,
                'overall_fitness': overall_fitness
            },
            'additional_metrics': {
                'final_ratio': paths["short"]["pheromone"] / paths["long"]["pheromone"] if paths["long"]["pheromone"] > 0 else float('inf'),
                'total_selections': len(path_selection_history["short"]) + len(path_selection_history["long"])
            },
            'final_pheromone_levels': paths,
            'total_path_selections': {
                'short': len(path_selection_history["short"]),
                'long': len(path_selection_history["long"])
            },
            'phase_selections': phase_analysis,
            'detailed_step_data': detailed_step_data,
            'pheromone_history': pheromone_history,
            'performance': {
                'start': start_perf,
                'end': end_perf,
                'cpu_change': end_perf['process_cpu_percent'] - start_perf['process_cpu_percent'],
                'memory_change_mb': end_perf['process_memory_mb'] - start_perf['process_memory_mb'],
                'avg_cpu': (start_perf['process_cpu_percent'] + end_perf['process_cpu_percent']) / 2,
                'avg_memory_mb': (start_perf['process_memory_mb'] + start_perf['process_memory_mb']) / 2,
                'max_memory_mb': max(start_perf['process_memory_mb'], end_perf['process_memory_mb']),
                'avg_threads': (start_perf['process_threads'] + end_perf['process_threads']) / 2,
                'avg_gpu_load': (start_perf['gpu_load_percent'] + end_perf['gpu_load_percent']) / 2,
                'avg_gpu_memory': (start_perf['gpu_memory_percent'] + end_perf['gpu_memory_percent']) / 2,
                'max_gpu_temp': max(start_perf['gpu_temperature'], end_perf['gpu_temperature'])
            }
        }
        
        overall_fitness = result['swarm_performance']['overall_fitness']
        print(f"✅ ACO LLM Simulation {run_number} completed successfully")
        print(f"   Duration: {simulation_duration:.1f}s | Steps: {final_step}/{max_iterations} | Overall Fitness: {overall_fitness:.3f}")
        
        return result
    
    def calculate_convergence_speed(self, short_selections, long_selections, max_iterations):
        """Measure how quickly the system converges to optimal path (higher = better)"""
        if len(short_selections) == 0 and len(long_selections) == 0:
            return 0.0
        
        # Find when system started preferring short path consistently
        convergence_point = max_iterations
        window_size = max(3, max_iterations // 10)  # Adaptive window
        
        for step in range(window_size, max_iterations):
            # Check if last window_size steps show strong preference for short path
            recent_short = len([s for s in short_selections if step - window_size <= s <= step])
            recent_long = len([s for s in long_selections if step - window_size <= s <= step])
            total_recent = recent_short + recent_long
            
            if total_recent > 0 and recent_short / total_recent >= 0.8:  # 80% preference
                convergence_point = step
                break
        
        # Earlier convergence = higher score (inverted and normalized)
        speed_score = max(0.0, 1.0 - (convergence_point / max_iterations))
        return speed_score
    
    def calculate_solution_quality(self, paths, optimal_path="short"):
        """Measure final solution quality - how well optimal path is reinforced (higher = better)"""
        optimal_pheromone = paths[optimal_path]["pheromone"]
        suboptimal_pheromone = sum(p["pheromone"] for k, p in paths.items() if k != optimal_path)
        
        if optimal_pheromone <= 0:
            return 0.0
        
        # Quality = ratio of optimal to suboptimal pheromone (normalized)
        if suboptimal_pheromone <= 0:
            return 1.0  # Perfect - only optimal path has pheromone
        
        ratio = optimal_pheromone / suboptimal_pheromone
        # Normalize using sigmoid to get 0-1 range (ratio of 10+ gives ~0.9+ quality)
        quality_score = ratio / (1 + ratio)
        return min(1.0, quality_score)
    
    def calculate_learning_progression(self, short_selections, long_selections, max_iterations):
        """Measure exploration, transition, and exploitation across 3 equal phases (higher = better)"""
        total_selections = len(short_selections) + len(long_selections)
        if total_selections == 0:
            return 0.0
        
        # Divide into 3 equal phases of 6 iterations each (or proportional)
        phase_size = max_iterations // 3
        
        # Phase 1: Exploration (iterations 1 to phase_size)
        exploration_short = len([s for s in short_selections if 1 <= s <= phase_size])
        exploration_long = len([s for s in long_selections if 1 <= s <= phase_size])
        exploration_total = exploration_short + exploration_long
        
        if exploration_total > 0:
            exploration_short_ratio = exploration_short / exploration_total
            # Perfect exploration = 50% short, 50% long (maximum diversity)
            exploration_score = 1.0 - abs(0.5 - exploration_short_ratio) * 2
        else:
            exploration_score = 0.0
        
        # Phase 2: Transition (iterations phase_size+1 to 2*phase_size)
        transition_short = len([s for s in short_selections if phase_size < s <= 2 * phase_size])
        transition_long = len([s for s in long_selections if phase_size < s <= 2 * phase_size])
        transition_total = transition_short + transition_long
        
        if transition_total > 0:
            transition_short_ratio = transition_short / transition_total
            # Good transition = gradual shift from 50% to 100% short, so ~75% is ideal
            target_transition = 0.75
            transition_score = 1.0 - abs(target_transition - transition_short_ratio) / target_transition
        else:
            transition_score = 0.0
        
        # Phase 3: Exploitation (iterations 2*phase_size+1 to max_iterations)
        exploitation_short = len([s for s in short_selections if s > 2 * phase_size])
        exploitation_long = len([s for s in long_selections if s > 2 * phase_size])
        exploitation_total = exploitation_short + exploitation_long
        
        if exploitation_total > 0:
            exploitation_short_ratio = exploitation_short / exploitation_total
            # Perfect exploitation = 100% short path selection
            exploitation_score = exploitation_short_ratio
        else:
            exploitation_score = 0.0
        
        # Average the three phase scores to get overall learning progression
        learning_progression = (exploration_score + transition_score + exploitation_score) / 3.0
        return min(1.0, max(0.0, learning_progression))
    
    def calculate_learning_stability(self, ratios):
        """Measure consistency of learning process (higher = better)"""
        if len(ratios) < 10:
            return 1.0  # Too few points to measure instability
        
        # Calculate how smooth the convergence is (less oscillation = better)
        if len(ratios) < 2:
            return 1.0
        
        # Measure trend consistency - good ACO should show monotonic improvement
        differences = np.diff(ratios)
        sign_changes = sum(1 for i in range(len(differences)-1) 
                          if differences[i] * differences[i+1] < 0)
        
        # Fewer sign changes = more stable (normalize by length)
        max_possible_changes = len(differences) - 1
        if max_possible_changes > 0:
            stability_score = 1.0 - (sign_changes / max_possible_changes)
        else:
            stability_score = 1.0
        
        return max(0.0, min(1.0, stability_score))
    
    def create_detailed_step_data_from_llm_results(self, pheromone_history, path_selection_history, final_paths, distance_weight, final_step):
        """Create detailed step data from LLM simulation results for visualizer compatibility"""
        detailed_step_data = []
        
        # Create timeline of selections
        all_selections = []
        for step in path_selection_history['short']:
            all_selections.append({'step': step, 'chosen': 'short'})
        for step in path_selection_history['long']:
            all_selections.append({'step': step, 'chosen': 'long'})
        
        # Sort by step
        all_selections.sort(key=lambda x: x['step'])
        
        # Create detailed tracking for each step
        steps_processed = min(final_step, len(pheromone_history))
        
        for step in range(steps_processed):
            # Get pheromone levels at this step
            if step < len(pheromone_history):
                pheromone_state = pheromone_history[step]
                short_pheromone = pheromone_state['short']
                long_pheromone = pheromone_state['long']
            else:
                # Use final state if needed
                short_pheromone = final_paths['short']['pheromone']
                long_pheromone = final_paths['long']['pheromone']
            
            # Calculate attractions (similar to regular ACO)
            short_distance = final_paths['short']['distance']
            long_distance = final_paths['long']['distance']
            
            short_attraction = short_pheromone / (short_distance ** distance_weight)
            long_attraction = long_pheromone / (long_distance ** distance_weight)
            total_attraction = short_attraction + long_attraction
            
            # Calculate probabilities
            short_prob = short_attraction / total_attraction if total_attraction > 0 else 0.5
            long_prob = long_attraction / total_attraction if total_attraction > 0 else 0.5
            
            # Calculate balance metric
            balance = min(short_prob, long_prob) * 2
            
            # Determine what was chosen at this step
            chosen = 'short'  # Default
            for selection in all_selections:
                if selection['step'] == step:
                    chosen = selection['chosen']
                    break
            
            # Store detailed step data
            detailed_step_data.append({
                'step': step,
                'short_pheromone': short_pheromone,
                'long_pheromone': long_pheromone,
                'short_prob': short_prob,
                'long_prob': long_prob,
                'balance': balance,
                'chosen': chosen,
                'short_attraction': short_attraction,
                'long_attraction': long_attraction
            })
        
        return detailed_step_data
    
    def calculate_phase_selections(self, detailed_step_data, max_iterations):
        """Calculate phase-based selection counts with dynamic phase sizing"""
        # Use all available step data for analysis
        analysis_steps = len(detailed_step_data)
        step_data_subset = detailed_step_data[:analysis_steps]
        
        # Calculate dynamic phase boundaries for equal division
        phase_size = analysis_steps // 3
        remainder = analysis_steps % 3
        
        # Distribute remainder across phases (early gets extra if remainder > 0, mid gets extra if remainder > 1)
        early_size = phase_size + (1 if remainder > 0 else 0)
        mid_size = phase_size + (1 if remainder > 1 else 0)
        late_size = phase_size
        
        # Calculate phase boundaries
        early_end = early_size - 1           # Steps 0 to early_end
        mid_end = early_size + mid_size - 1  # Steps early_size to mid_end
                                            # Steps mid_end+1 to analysis_steps-1
        
        # Count selections per phase
        early_selections = [d['chosen'] for d in step_data_subset if d['step'] <= early_end]
        mid_selections = [d['chosen'] for d in step_data_subset if early_end < d['step'] <= mid_end]
        late_selections = [d['chosen'] for d in step_data_subset if d['step'] > mid_end]
        
        # Calculate counts
        phase_data = {
            'early_phase': {
                'steps': f"0-{early_end}",
                'total_steps': len(early_selections),
                'short_count': sum(1 for s in early_selections if s == 'short'),
                'long_count': sum(1 for s in early_selections if s == 'long'),
                'short_percentage': (sum(1 for s in early_selections if s == 'short') / len(early_selections) * 100) if early_selections else 0,
                'selections': early_selections
            },
            'mid_phase': {
                'steps': f"{early_end+1}-{mid_end}",
                'total_steps': len(mid_selections),
                'short_count': sum(1 for s in mid_selections if s == 'short'),
                'long_count': sum(1 for s in mid_selections if s == 'long'),
                'short_percentage': (sum(1 for s in mid_selections if s == 'short') / len(mid_selections) * 100) if mid_selections else 0,
                'selections': mid_selections
            },
            'late_phase': {
                'steps': f"{mid_end+1}-{analysis_steps-1}",
                'total_steps': len(late_selections),
                'short_count': sum(1 for s in late_selections if s == 'short'),
                'long_count': sum(1 for s in late_selections if s == 'long'),
                'short_percentage': (sum(1 for s in late_selections if s == 'short') / len(late_selections) * 100) if late_selections else 0,
                'selections': late_selections
            },
            'analysis_info': {
                'total_steps_analyzed': analysis_steps,
                'early_phase_size': early_size,
                'mid_phase_size': mid_size,
                'late_phase_size': late_size,
                'phase_distribution': f"{early_size}-{mid_size}-{late_size}"
            }
        }
        
        # Add phase progression summary
        phase_data['phase_progression'] = f"{phase_data['early_phase']['short_percentage']:.1f}% → {phase_data['mid_phase']['short_percentage']:.1f}% → {phase_data['late_phase']['short_percentage']:.1f}%"
        
        return phase_data
    
    def run_multiple_trials(self):
        """Run multiple ACO LLM trials with different seeds"""
        print(f"\n🚀 Starting Multi-Run ACO LLM Analysis")
        print(f"{'='*60}")
        print(f"Number of trials: {len(self.seeds)}")
        print(f"Max iterations per trial: {self.config['max_iterations']}")
        if len(self.seeds) <= 10:
            print(f"Seeds: {self.seeds}")
        else:
            print(f"Seeds: {self.seeds[:5]} ... {self.seeds[-2:]} (showing first 5 and last 2)")
        print(f"{'='*60}\n")
        
        # Record total analysis start time
        self.total_start_time = time.time()
        trial_times = []
        
        for i, seed in enumerate(self.seeds):
            try:
                trial_start = time.time()
                result = self.run_single_aco_llm(seed, i + 1)
                trial_end = time.time()
                trial_duration = trial_end - trial_start
                trial_times.append(trial_duration)
                
                # Add trial timing to result
                result['trial_timing'] = {
                    'duration_seconds': trial_duration,
                    'start_timestamp': trial_start,
                    'end_timestamp': trial_end
                }
                
                self.results.append(result)
            except Exception as e:
                print(f"❌ Error in simulation {i+1} with seed {seed}: {e}")
                continue
        
        # Record total analysis end time
        self.total_end_time = time.time()
        total_duration = self.total_end_time - self.total_start_time
        
        # Calculate timing statistics
        if trial_times:
            self.timing_statistics = {
                'total_duration_seconds': total_duration,
                'total_duration_minutes': total_duration / 60,
                'average_trial_duration': np.mean(trial_times),
                'shortest_trial_duration': np.min(trial_times),
                'longest_trial_duration': np.max(trial_times),
                'trial_duration_std': np.std(trial_times),
                'trials_per_minute': len(trial_times) / (total_duration / 60) if total_duration > 0 else 0
            }
        
        print(f"\n🎯 Multi-Run Analysis Complete!")
        print(f"{'='*60}")
        print(f"⏱️  TIMING SUMMARY:")
        print(f"Total time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"Successful trials: {len(self.results)}/{len(self.seeds)}")
        print(f"Average trial duration: {np.mean(trial_times):.2f} seconds")
        print(f"Shortest trial: {np.min(trial_times):.2f} seconds")
        print(f"Longest trial: {np.max(trial_times):.2f} seconds")
        print(f"Trials per minute: {len(trial_times) / (total_duration / 60):.2f}")
        print(f"{'='*60}")
    
    def calculate_statistics(self):
        """Calculate mean, std, min, max for all metrics"""
        if not self.results:
            return {}
        
        stats = {
            'swarm_stats': {},
            'additional_stats': {},
            'timing_stats': {}
        }
        
        # Swarm performance statistics
        for metric in ['convergence_speed', 'solution_quality', 'learning_efficiency', 'learning_stability', 'overall_fitness']:
            values = [r['swarm_performance'][metric] for r in self.results]
            stats['swarm_stats'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # Additional metrics statistics
        for metric in ['final_ratio']:
            values = [r['additional_metrics'][metric] for r in self.results if r['additional_metrics'][metric] != float('inf')]
            if values:
                stats['additional_stats'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Timing statistics
        timing_values = [r['convergence_time_seconds'] for r in self.results]
        step_values = [r['final_step'] for r in self.results]
        trial_timing_values = [r['trial_timing']['duration_seconds'] for r in self.results if 'trial_timing' in r]
        
        stats['timing_stats'] = {
            'convergence_time': {
                'mean': float(np.mean(timing_values)),
                'std': float(np.std(timing_values)),
                'min': float(np.min(timing_values)),
                'max': float(np.max(timing_values))
            },
            'steps_to_convergence': {
                'mean': float(np.mean(step_values)),
                'std': float(np.std(step_values)),
                'min': float(np.min(step_values)),
                'max': float(np.max(step_values))
            }
        }
        
        # Add total timing statistics if available
        if hasattr(self, 'timing_statistics'):
            stats['overall_timing'] = self.timing_statistics
        
        # Performance statistics (process-specific)
        cpu_values = [r['performance']['avg_cpu'] for r in self.results if 'performance' in r]
        memory_values = [r['performance']['avg_memory_mb'] for r in self.results if 'performance' in r]
        max_memory_values = [r['performance']['max_memory_mb'] for r in self.results if 'performance' in r]
        thread_values = [r['performance']['avg_threads'] for r in self.results if 'performance' in r]
        gpu_load_values = [r['performance']['avg_gpu_load'] for r in self.results if 'performance' in r]
        gpu_memory_values = [r['performance']['avg_gpu_memory'] for r in self.results if 'performance' in r]
        
        if cpu_values:
            stats['performance_stats'] = {
                'process_cpu_usage': {
                    'mean': float(np.mean(cpu_values)),
                    'std': float(np.std(cpu_values)),
                    'min': float(np.min(cpu_values)),
                    'max': float(np.max(cpu_values))
                },
                'process_memory_usage_mb': {
                    'mean': float(np.mean(memory_values)),
                    'std': float(np.std(memory_values)),
                    'min': float(np.min(memory_values)),
                    'max': float(np.max(memory_values))
                },
                'max_memory_usage_mb': {
                    'mean': float(np.mean(max_memory_values)),
                    'std': float(np.std(max_memory_values)),
                    'min': float(np.min(max_memory_values)),
                    'max': float(np.max(max_memory_values))
                },
                'process_threads': {
                    'mean': float(np.mean(thread_values)),
                    'std': float(np.std(thread_values)),
                    'min': float(np.min(thread_values)),
                    'max': float(np.max(thread_values))
                }
            }
            
            if gpu_load_values and any(v > 0 for v in gpu_load_values):
                stats['performance_stats']['gpu_load_percent'] = {
                    'mean': float(np.mean(gpu_load_values)),
                    'std': float(np.std(gpu_load_values)),
                    'min': float(np.min(gpu_load_values)),
                    'max': float(np.max(gpu_load_values))
                }
                stats['performance_stats']['gpu_memory_percent'] = {
                    'mean': float(np.mean(gpu_memory_values)),
                    'std': float(np.std(gpu_memory_values)),
                    'min': float(np.min(gpu_memory_values)),
                    'max': float(np.max(gpu_memory_values))
                }
        
        return stats
    
    def create_multi_run_visualization(self):
        """Create 4-panel visualization for multi-run analysis"""
        if not self.results:
            print("❌ No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'ACO LLM Multi-Run Analysis (n={len(self.results)})', fontsize=16, fontweight='bold')
        
        # Panel 1: Swarm Performance Metrics (means with error bars)
        ax1 = axes[0, 0]
        stats = self.calculate_statistics()
        
        metrics = ['convergence_speed', 'solution_quality', 'learning_efficiency', 'learning_stability', 'overall_fitness']
        metric_labels = ['Convergence\nSpeed', 'Solution\nQuality', 'Learning\nEfficiency', 'Learning\nStability', 'Overall\nFitness']
        means = [stats['swarm_stats'][metric]['mean'] for metric in metrics]
        stds = [stats['swarm_stats'][metric]['std'] for metric in metrics]
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightcoral']
        
        bars = ax1.bar(metric_labels, means, yerr=stds, color=colors, alpha=0.7, 
                      edgecolor='black', capsize=5)
        ax1.set_title('ACO LLM Performance Metrics', fontweight='bold')
        ax1.set_ylabel('Score (0-1)')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel 2: Performance Variability (Standard Deviation)
        ax2 = axes[0, 1]
        bars2 = ax2.bar(metric_labels, stds, color='gold', alpha=0.7, edgecolor='black')
        ax2.set_title('Performance Variability (Standard Deviation)', fontweight='bold')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, std in zip(bars2, stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(stds)*0.01,
                    f'{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel 3: Convergence Analysis
        ax3 = axes[1, 0]
        convergence_times = [r['convergence_time_seconds'] for r in self.results]
        final_steps = [r['final_step'] for r in self.results]
        
        # Scatter plot of convergence time vs steps
        scatter = ax3.scatter(final_steps, convergence_times, alpha=0.6, s=50, c='purple')
        ax3.set_xlabel('Steps to Convergence')
        ax3.set_ylabel('Convergence Time (seconds)')
        ax3.set_title('Convergence Analysis', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        if len(final_steps) > 1:
            z = np.polyfit(final_steps, convergence_times, 1)
            p = np.poly1d(z)
            ax3.plot(final_steps, p(final_steps), "r--", alpha=0.8)
        
        # Panel 4: Overall Fitness by Run
        ax4 = axes[1, 1]
        fitness_values = [r['swarm_performance']['overall_fitness'] for r in self.results]
        run_numbers = range(1, len(fitness_values) + 1)
        
        ax4.plot(run_numbers, fitness_values, 'mo-', alpha=0.7, linewidth=1, markersize=4)
        ax4.axhline(y=np.mean(fitness_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(fitness_values):.3f}')
        ax4.set_xlabel('Run Number')
        ax4.set_ylabel('Overall Fitness')
        ax4.set_title('Overall Fitness by Run', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_results(self):
        """Save comprehensive results to JSON file"""
        if not self.results:
            print("❌ No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare comprehensive results
        comprehensive_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config_file': self.config_file,
                'seeds_used': self.seeds,
                'successful_runs': len(self.results),
                'total_attempted': len(self.seeds),
                'analysis_type': 'ACO_LLM_Multi_Run'
            },
            'statistics': self.calculate_statistics(),
            'individual_results': self.results,
            'configuration': self.config
        }
        
        # Convert NumPy types to native Python types
        comprehensive_results = convert_numpy_types(comprehensive_results)
        
        # Save results
        results_dir = "results/aco_llm"
        os.makedirs(results_dir, exist_ok=True)
        filename = f"{results_dir}/multi_run_aco_llm_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print comprehensive summary of results"""
        if not self.results:
            print("❌ No results to summarize")
            return
        
        stats = self.calculate_statistics()
        
        print(f"\n{'='*80}")
        print("                  MULTI-RUN ACO LLM ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Simulations completed: {len(self.results)}")
        print(f"Configuration: {self.config['max_iterations']} max iterations, target ratio {self.config['target_ratio']}")
        print(f"Seeds used: {[r['seed'] for r in self.results]}")
        
        print(f"\n📊 ACO LLM Performance Statistics:")
        print(f"{'Metric':<25} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print(f"{'-'*60}")
        
        for metric, data in stats['swarm_stats'].items():
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<25} {data['mean']:<8.3f} {data['std']:<8.3f} "
                  f"{data['min']:<8.3f} {data['max']:<8.3f}")
        
        print(f"\n⏱️  Timing Statistics:")
        timing_stats = stats['timing_stats']
        print(f"Simulation Time:          {timing_stats['convergence_time']['mean']:.2f}s ± {timing_stats['convergence_time']['std']:.2f}s")
        print(f"Steps to Convergence:     {timing_stats['steps_to_convergence']['mean']:.1f} ± {timing_stats['steps_to_convergence']['std']:.1f}")
        
        if 'overall_timing' in stats:
            ot = stats['overall_timing']
            print(f"\n⏰ Overall Analysis Timing:")
            print(f"Total analysis time:      {ot['total_duration_seconds']:.2f}s ({ot['total_duration_minutes']:.2f} min)")
            print(f"Average trial duration:   {ot['average_trial_duration']:.2f}s")
            print(f"Shortest trial:           {ot['shortest_trial_duration']:.2f}s")
            print(f"Longest trial:            {ot['longest_trial_duration']:.2f}s")
            print(f"Trials per minute:        {ot['trials_per_minute']:.2f}")
        
        if 'performance_stats' in stats:
            perf_stats = stats['performance_stats']
            print(f"\n💻 Process Performance Statistics:")
            print(f"CPU Usage (process):      {perf_stats['process_cpu_usage']['mean']:.2f}% ± {perf_stats['process_cpu_usage']['std']:.2f}%")
            print(f"Memory Usage (process):   {perf_stats['process_memory_usage_mb']['mean']:.1f}MB ± {perf_stats['process_memory_usage_mb']['std']:.1f}MB")
            print(f"Peak Memory (process):    {perf_stats['max_memory_usage_mb']['max']:.1f}MB")
            print(f"Average Threads:          {perf_stats['process_threads']['mean']:.1f}")
            
            if 'gpu_load_percent' in perf_stats:
                print(f"GPU Load:                 {perf_stats['gpu_load_percent']['mean']:.1f}% ± {perf_stats['gpu_load_percent']['std']:.1f}%")
                print(f"GPU Memory:               {perf_stats['gpu_memory_percent']['mean']:.1f}% ± {perf_stats['gpu_memory_percent']['std']:.1f}%")
        
        if 'final_ratio' in stats['additional_stats']:
            ratio_stats = stats['additional_stats']['final_ratio']
            print(f"\n📈 Final Ratio Statistics:")
            print(f"Final Ratio:              {ratio_stats['mean']:.2f} ± {ratio_stats['std']:.2f}")
        
        # Show sample phase analysis from first result
        if self.results and 'phase_selections' in self.results[0]:
            print(f"\n📊 Phase Selection Analysis (Sample from Run 1):")
            phase_data = self.results[0]['phase_selections']
            print(f"Phase Progression:        {phase_data['phase_progression']}")
            print(f"Early Phase ({phase_data['early_phase']['steps']}):        {phase_data['early_phase']['short_count']}S/{phase_data['early_phase']['long_count']}L ({phase_data['early_phase']['short_percentage']:.1f}% short)")
            print(f"Mid Phase ({phase_data['mid_phase']['steps']}):          {phase_data['mid_phase']['short_count']}S/{phase_data['mid_phase']['long_count']}L ({phase_data['mid_phase']['short_percentage']:.1f}% short)")
            print(f"Late Phase ({phase_data['late_phase']['steps']}):       {phase_data['late_phase']['short_count']}S/{phase_data['late_phase']['long_count']}L ({phase_data['late_phase']['short_percentage']:.1f}% short)")
            
            # Show average phase progressions across all runs
            if len(self.results) > 1:
                early_percentages = [r['phase_selections']['early_phase']['short_percentage'] for r in self.results]
                mid_percentages = [r['phase_selections']['mid_phase']['short_percentage'] for r in self.results]
                late_percentages = [r['phase_selections']['late_phase']['short_percentage'] for r in self.results]
                
                print(f"\n📊 Average Phase Performance Across All {len(self.results)} Runs:")
                print(f"Early Phase Average:      {np.mean(early_percentages):.1f}% ± {np.std(early_percentages):.1f}% short selections")
                print(f"Mid Phase Average:        {np.mean(mid_percentages):.1f}% ± {np.std(mid_percentages):.1f}% short selections")  
                print(f"Late Phase Average:       {np.mean(late_percentages):.1f}% ± {np.std(late_percentages):.1f}% short selections")
                print(f"Average Progression:      {np.mean(early_percentages):.1f}% → {np.mean(mid_percentages):.1f}% → {np.mean(late_percentages):.1f}%")

def main():
    """Main function to run multi-trial ACO LLM analysis"""
    print("🤖 Multi-Run ACO LLM Analysis")
    print("=" * 50)
    
    try:
        # Create analyzer
        analyzer = MultiRunACOLLMAnalyzer()
        
        # Run multiple trials
        analyzer.run_multiple_trials()
        
        # Print summary
        analyzer.print_summary()
        
        # Save results
        results_file = analyzer.save_results()
        
        # Create and save visualization
        fig = analyzer.create_multi_run_visualization()
        if fig:
            viz_dir = "results/visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_filename = f'{viz_dir}/aco_llm_multi_run_{timestamp}.png'
            fig.savefig(viz_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {viz_filename}")
            plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
