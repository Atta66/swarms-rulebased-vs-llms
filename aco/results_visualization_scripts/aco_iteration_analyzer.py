"""
ACO Iteration Analyzer - Latest Multi_Run Results
Analyzes the latest multi_run ACO results and shows how the average short path percentage 
changes over iterations, providing comprehensive analysis of learning progression.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob

def find_500_iteration_aco_results():
    """Find the latest multi_run ACO results file."""
    results_dir = "../results/aco"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory {results_dir} not found!")
        return None
    
    # Target the latest multi_run files (priority order)
    target_files = [
        os.path.join(results_dir, "multi_run_aco_20251020_164251.json"),
        os.path.join(results_dir, "multi_run_aco_20251020_163353.json"),
        os.path.join(results_dir, "multi_run_aco_20251020_162345.json"),
        os.path.join(results_dir, "multi_run_aco_20251020_162112.json")
    ]
    
    for target_file in target_files:
        if os.path.exists(target_file):
            print(f"📊 Using latest multi_run ACO results: {os.path.basename(target_file)}")
            return target_file
    
    # If no latest files found, search for any multi_run files (most recent first)
    print("🔍 Searching for any multi_run ACO results...")
    pattern = os.path.join(results_dir, "multi_run_aco_*.json")
    files = sorted(glob.glob(pattern), reverse=True)  # Most recent first
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                # Try to load the file to verify it's valid
                data = json.load(f)
                print(f"✅ Found multi_run ACO results: {os.path.basename(file_path)}")
                return file_path
        except:
            continue
    
    print("❌ No valid multi_run ACO results found!")
    return None

def load_and_analyze_results(file_path):
    """Load ACO results and analyze iteration-by-iteration progression."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded results file")
        
        # Check the actual structure of the data
        print(f"📋 Available keys: {list(data.keys())}")
        
        # Extract configuration info
        config = data.get('configuration', {})
        max_iterations = config.get('max_iterations', 'Unknown')
        
        # Show iteration count
        print(f"📊 File contains {max_iterations} iterations")
        
        # Look for individual results in different possible locations
        individual_results = None
        for key in ['individual_results', 'results', 'trials', 'data']:
            if key in data:
                individual_results = data[key]
                print(f"📊 Found individual results in '{key}' with {len(individual_results)} trials")
                break
        
        if individual_results is None:
            # Check if the data itself is a list
            if isinstance(data, list):
                individual_results = data
                print(f"📊 Data is directly a list with {len(individual_results)} trials")
            else:
                print(f"❌ Could not find individual results in the data structure")
                print(f"Available top-level keys: {list(data.keys())}")
                return None, None
        
        num_trials = len(individual_results)
        print(f"📋 Configuration: {max_iterations} iterations, {num_trials} trials")
        
        # Analyze step-by-step progression
        all_iterations_data = []
        
        for trial_idx, trial in enumerate(individual_results):
            # Look for detailed step data in different possible locations
            trial_data = None
            for key in ['detailed_step_data', 'step_data', 'steps', 'iterations']:
                if key in trial:
                    trial_data = trial[key]
                    break
            
            if trial_data is None:
                print(f"⚠️  Trial {trial_idx + 1}: No step data found")
                if trial_idx == 0:  # Show structure of first trial
                    print(f"   Available keys in trial: {list(trial.keys())}")
                continue
            
            for step_idx, step in enumerate(trial_data):
                iteration = step_idx + 1  # 1-based iteration numbering
                
                # Calculate short path percentage for this step
                # Look for the path choice information
                chosen = step.get('chosen', None)
                
                if chosen is not None:
                    # Count this selection
                    if chosen == "short":
                        short_percentage = 100.0  # This iteration chose short
                    else:
                        short_percentage = 0.0    # This iteration chose long
                else:
                    # Try alternative data structures
                    short_selections = step.get('short_selections', 0)
                    total_selections = step.get('total_ants', step.get('total_selections', 1))
                    
                    if total_selections > 0:
                        short_percentage = (short_selections / total_selections) * 100
                    else:
                        short_percentage = 0
                
                all_iterations_data.append({
                    'trial': trial_idx + 1,
                    'iteration': iteration,
                    'short_percentage': short_percentage,
                    'chosen': chosen
                })
        
        if not all_iterations_data:
            print("❌ No iteration data could be extracted")
            return None, None
            
        print(f"📊 Successfully extracted {len(all_iterations_data)} data points")
        return all_iterations_data, config
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def calculate_iteration_averages(iterations_data, max_iterations):
    """Calculate average short percentage for each iteration across all trials."""
    iteration_averages = {}
    
    # Group data by iteration
    for data_point in iterations_data:
        iteration = data_point['iteration']
        short_pct = data_point['short_percentage']
        
        if iteration not in iteration_averages:
            iteration_averages[iteration] = []
        
        iteration_averages[iteration].append(short_pct)
    
    # Calculate averages and standard deviations
    iterations = []
    averages = []
    std_devs = []
    
    # Only process iterations that we have data for
    available_iterations = sorted(iteration_averages.keys())
    max_available = max(available_iterations) if available_iterations else 0
    
    print(f"📊 Available iterations: 1 to {max_available}")
    
    for iteration in range(1, min(max_iterations, max_available) + 1):
        if iteration in iteration_averages:
            values = iteration_averages[iteration]
            avg = np.mean(values)
            std = np.std(values)
            
            iterations.append(iteration)
            averages.append(avg)
            std_devs.append(std)
    
    print(f"📈 Processed {len(iterations)} iterations with data")
    return iterations, averages, std_devs

def smooth_data(iterations, averages, std_devs, window_size=10):
    """Apply moving average to smooth the data for better readability."""
    if len(averages) == 0:
        return iterations, averages, std_devs
    
    # Apply moving average
    smoothed_averages = []
    smoothed_std_devs = []
    smoothed_iterations = []
    
    for i in range(len(averages)):
        # Define window boundaries
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(averages), i + window_size // 2 + 1)
        
        # Calculate moving average
        window_avgs = averages[start_idx:end_idx]
        window_stds = std_devs[start_idx:end_idx]
        
        smoothed_avg = np.mean(window_avgs)
        smoothed_std = np.mean(window_stds)  # Average the standard deviations
        
        smoothed_averages.append(smoothed_avg)
        smoothed_std_devs.append(smoothed_std)
        smoothed_iterations.append(iterations[i])
    
    return smoothed_iterations, smoothed_averages, smoothed_std_devs

def create_progression_graph(iterations, averages, std_devs, config):
    """Create a graph showing the progression of short path percentage over iterations."""
    
    # Convert to numpy arrays for easier manipulation
    iterations = np.array(iterations)
    averages = np.array(averages)
    std_devs = np.array(std_devs)
    
    # Apply smoothing for better readability
    smooth_iter, smooth_avg, smooth_std = smooth_data(iterations, averages, std_devs, window_size=20)
    smooth_iter = np.array(smooth_iter)
    smooth_avg = np.array(smooth_avg)
    smooth_std = np.array(smooth_std)
    
    # Create single plot
    plt.figure(figsize=(16, 8))
    
    # Main plot with only smoothed data
    plt.plot(smooth_iter, smooth_avg, 'darkblue', linewidth=3, label='Average Optimal Path Selection')
    
    # Add confidence interval for smoothed data (clamp to 0-100% range)
    lower_bound = np.maximum(smooth_avg - smooth_std, 0)  # Don't go below 0%
    upper_bound = np.minimum(smooth_avg + smooth_std, 100)  # Don't go above 100%
    
    plt.fill_between(smooth_iter, 
                     lower_bound, 
                     upper_bound, 
                     alpha=0.2, color='darkblue', label='±1 Standard Deviation (Trial Variability)')
    
    # Formatting
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Optimal Path Selection (%)', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aco_latest_multirun_progression_{timestamp}.png"
    save_path = os.path.join("..", "results", "visualizations", filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Latest multi_run graph saved to: {save_path}")
    
    plt.show()
    
    return save_path

def print_phase_analysis(iterations, averages, max_iterations):
    """Print analysis of different phases of learning."""
    if len(averages) == 0:
        return
    
    # Define phases dynamically based on actual max_iterations
    early_end = max_iterations // 3  # First third
    mid_end = 2 * max_iterations // 3  # Middle third
    
    early_phase = [avg for i, avg in enumerate(averages) if iterations[i] <= early_end]
    mid_phase = [avg for i, avg in enumerate(averages) if early_end < iterations[i] <= mid_end]
    late_phase = [avg for i, avg in enumerate(averages) if iterations[i] > mid_end]
    
    print("\n" + "="*60)
    print(f"📊 PHASE ANALYSIS ({max_iterations} ITERATIONS)")
    print("="*60)
    
    if early_phase:
        print(f"🔍 Early Phase (1-{early_end}): {np.mean(early_phase):.1f}% ± {np.std(early_phase):.1f}%")
    
    if mid_phase:
        print(f"🎯 Mid Phase ({early_end+1}-{mid_end}): {np.mean(mid_phase):.1f}% ± {np.std(mid_phase):.1f}%")
    
    if late_phase:
        print(f"🚀 Late Phase ({mid_end+1}-{max_iterations}): {np.mean(late_phase):.1f}% ± {np.std(late_phase):.1f}%")
    
    # Learning trend
    if len(averages) >= 2:
        learning_rate = (averages[-1] - averages[0]) / len(averages)
        print(f"📈 Learning Rate: {learning_rate:.3f}% per iteration")
    
    # Convergence analysis for any number of iterations
    if len(averages) >= 10:  # Need at least 10 iterations for meaningful analysis
        convergence_80 = None
        convergence_90 = None
        convergence_95 = None
        
        for i, avg in enumerate(averages):
            if convergence_80 is None and avg >= 80:
                convergence_80 = iterations[i]
            if convergence_90 is None and avg >= 90:
                convergence_90 = iterations[i]
            if convergence_95 is None and avg >= 95:
                convergence_95 = iterations[i]
        
        print(f"\n🎯 Convergence Analysis:")
        if convergence_80:
            print(f"   80% accuracy reached at iteration: {convergence_80}")
        if convergence_90:
            print(f"   90% accuracy reached at iteration: {convergence_90}")
        if convergence_95:
            print(f"   95% accuracy reached at iteration: {convergence_95}")

def main():
    """Main analysis function."""
    print("🐜 Latest Multi_Run ACO Analysis")
    print("="*50)
    
    # Find and load the latest multi_run results
    target_file = find_500_iteration_aco_results()
    if not target_file:
        return
    
    # Load and analyze the data
    iterations_data, config = load_and_analyze_results(target_file)
    if not iterations_data:
        return
    
    max_iterations = config.get('max_iterations', 500)
    print(f"📊 Analyzing {len(iterations_data)} data points across {max_iterations} iterations")
    
    # Calculate iteration averages
    iterations, averages, std_devs = calculate_iteration_averages(iterations_data, max_iterations)
    
    # Create the progression graph with smoothing
    save_path = create_progression_graph(iterations, averages, std_devs, config)
    
    # Print phase analysis
    print_phase_analysis(iterations, averages, max_iterations)
    
    print(f"\n✅ Latest multi_run analysis complete! Graph saved to: {save_path}")

if __name__ == "__main__":
    main()
