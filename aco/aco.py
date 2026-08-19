import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import GPUtil
import json
import os
from datetime import datetime

# Load configuration
def load_config(config_file="config/aco_config.json"):
    """Load ACO configuration from JSON file"""
    if not os.path.exists(config_file):
        print(f"⚠️  Config file {config_file} not found, using defaults")
        return {
            "max_iterations": 1000,
            "evaporation_rate": 0.01,
            "target_ratio": 236.2,
            "ratio_tolerance": 1.0,
            "early_steps_for_exploration": 100,
            "paths": {
                "short": {"distance": 1, "initial_pheromone": 1},
                "long": {"distance": 2, "initial_pheromone": 1}
            },
            "visualization": {"show_plots": True, "pause_duration": 0.2},
            "performance": {"save_results": True, "results_directory": "results/aco"}
        }
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(f"📋 Loaded configuration from {config_file}")
    return config

# Load configuration
config = load_config()

# Setup for resource monitoring
cpu_log = []
ram_log = []
gpu_load_log = []
gpu_mem_log = []
timestamps = []

process = psutil.Process()
start_time = time.time()

# Configuration values
max_iterations = config["max_iterations"]
evaporation_rate = config["evaporation_rate"]
target_ratio = config["target_ratio"]
ratio_tolerance = config["ratio_tolerance"]
early_steps = config["early_steps_for_exploration"]

print(f"🐜 Starting ACO with {max_iterations} max iterations")
print(f"📊 Target ratio: {target_ratio} ± {ratio_tolerance}")
print(f"💨 Evaporation rate: {evaporation_rate}")

# Performance tracking
pheromone_history = []
path_selection_history = {"short": [], "long": []}
step_wise_ratios = []

# Two possible paths between A and B (from config)
paths = {
    "short": {
        "distance": config["paths"]["short"]["distance"], 
        "pheromone": config["paths"]["short"]["initial_pheromone"]
    },
    "long": {
        "distance": config["paths"]["long"]["distance"], 
        "pheromone": config["paths"]["long"]["initial_pheromone"]
    }
}

def choose_path(paths):
    total = sum(p["pheromone"] / p["distance"] for p in paths.values())
    probs = {name: (p["pheromone"] / p["distance"]) / total for name, p in paths.items()}
    return np.random.choice(list(probs.keys()), p=list(probs.values()))

def calculate_exploration_efficiency(short_selections, long_selections, early_steps=100):
    """Measure path diversity in early steps (0-1 scale)"""
    if len(short_selections) < early_steps:
        early_short = len([x for x in short_selections if x < early_steps])
        early_long = len([x for x in long_selections if x < early_steps])
    else:
        early_short = len([x for x in short_selections[:early_steps] if x < early_steps])
        early_long = len([x for x in long_selections[:early_steps] if x < early_steps])
    
    total_early = early_short + early_long
    if total_early == 0:
        return 0.5  # No selections yet
    
    # Good exploration = balanced path usage initially
    short_ratio = early_short / total_early
    # Perfect exploration = 0.5 (equal usage), worst = 0 or 1 (only one path)
    exploration_score = 1.0 - abs(0.5 - short_ratio) * 2
    return max(0.0, exploration_score)

def calculate_convergence_quality(paths, optimal_path="short"):
    """Measure how strongly the optimal path dominates (0-1 scale)"""
    optimal_pheromone = paths[optimal_path]["pheromone"]
    total_pheromone = sum(p["pheromone"] for p in paths.values())
    
    if total_pheromone == 0:
        return 0
    
    # Higher concentration on optimal path = better convergence
    convergence_score = optimal_pheromone / total_pheromone
    return min(1.0, convergence_score)

def calculate_learning_stability(ratios):
    """Measure consistency of pheromone ratio changes (0-1 scale)"""
    if len(ratios) < 10:
        return 1.0
    
    # Calculate variance in pheromone ratios
    if len(ratios) < 2:
        return 1.0
    
    # Lower variance in learning = higher stability
    ratio_variance = np.var(ratios)
    stability_score = 1.0 / (1.0 + ratio_variance)
    return min(1.0, stability_score)

def calculate_convergence_speed(step_count, max_steps):
    """How quickly the system reaches optimal ratio (0-1 scale, higher=faster)"""
    speed_score = max(0, 1.0 - (step_count / max_steps))
    return speed_score

def calculate_resource_efficiency(cpu_log, ram_log, gpu_log):
    """How efficiently the algorithm uses computational resources (0-1 scale)"""
    if not cpu_log:
        return 1.0
    
    avg_cpu = np.mean(cpu_log)
    avg_ram = np.mean(ram_log) 
    avg_gpu = np.mean(gpu_log) if gpu_log else 0
    
    # Lower resource usage = higher efficiency (normalize by expected max usage)
    avg_usage = (avg_cpu + avg_ram + avg_gpu) / 3
    efficiency_score = max(0, 1.0 - (avg_usage / 100))
    return efficiency_score

def calculate_overall_fitness(exploration, convergence, stability):
    """Combined ACO performance metric"""
    return (exploration + convergence + stability) / 3.0

# Simulate ants
history = {"short": 0, "long": 0}
for step in range(max_iterations):
    current_time = time.time() - start_time
    timestamps.append(current_time)

    # Resource usage
    cpu_log.append(psutil.cpu_percent(interval=None))
    ram_log.append(psutil.virtual_memory().percent)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_log = gpus[0]
        gpu_load_log.append(gpu_log.load * 100)
        gpu_mem_log.append(gpu_log.memoryUtil * 100)
    else:
        gpu_load_log.append(0)
        gpu_mem_log.append(0)

    # ACO logic
    chosen = choose_path(paths)
    history[chosen] += 1
    
    # Track selections for analysis
    if chosen == "short":
        path_selection_history["short"].append(step)
    else:
        path_selection_history["long"].append(step)
    
    # Update pheromones
    paths[chosen]["pheromone"] += 1 / paths[chosen]["distance"]
    for path in paths.values():
        path["pheromone"] *= (1 - evaporation_rate)
    
    # Store pheromone state for analysis
    pheromone_history.append({
        "short": paths["short"]["pheromone"],
        "long": paths["long"]["pheromone"],
        "step": step
    })
    
    # Calculate and store ratio
    if paths["long"]["pheromone"] > 0:
        ratio = paths["short"]["pheromone"] / paths["long"]["pheromone"]
        step_wise_ratios.append(ratio)
    else:
        step_wise_ratios.append(float('inf'))

    # Plot the pheromone levels (if visualization enabled)
    if config["visualization"]["show_plots"]:
        plt.clf()
        plt.bar(paths.keys(), [p["pheromone"] for p in paths.values()], color=['green', 'blue'])
        plt.title(f"Step {step+1}/{max_iterations}")
        plt.ylabel("Pheromone Level")
        plt.pause(config["visualization"]["pause_duration"])

    # Stop if short/long pheromone ratio is close to target
    ratio = paths["short"]["pheromone"] / paths["long"]["pheromone"]
    if abs(ratio - target_ratio) < ratio_tolerance:
        if step % 100 == 0:
            print(f"Step {step+1}: Short pheromone = {paths['short']['pheromone']:.2f}, Long pheromone = {paths['long']['pheromone']:.2f}, Ratio = {ratio:.2f}")
        print(f"🎯 Stopping early: short/long pheromone ratio reached {ratio:.2f} at step {step+1}")
        final_step = step + 1
        break
else:
    final_step = max_iterations
    print(f"⏰ Reached maximum iterations ({max_iterations})")

# Final timing and reporting
end_time = time.time()
convergence_time = end_time - start_time

# Calculate performance metrics
exploration_efficiency = calculate_exploration_efficiency(
    path_selection_history["short"], 
    path_selection_history["long"]
)
convergence_quality = calculate_convergence_quality(paths)
learning_stability = calculate_learning_stability(step_wise_ratios)
convergence_speed = calculate_convergence_speed(final_step, max_iterations)
resource_efficiency = calculate_resource_efficiency(cpu_log, ram_log, gpu_load_log)
overall_fitness = calculate_overall_fitness(exploration_efficiency, convergence_quality, learning_stability)

# Performance results
performance_results = {
    "timestamp": datetime.now().isoformat(),
    "configuration": {
        "max_iterations": max_iterations,
        "evaporation_rate": evaporation_rate,
        "target_ratio": target_ratio,
        "ratio_tolerance": ratio_tolerance
    },
    "convergence_time_seconds": convergence_time,
    "final_step": final_step,
    "swarm_performance": {
        "exploration_efficiency": exploration_efficiency,
        "convergence_quality": convergence_quality,
        "learning_stability": learning_stability,
        "overall_fitness": overall_fitness
    },
    "additional_metrics": {
        "convergence_speed": convergence_speed,
        "resource_efficiency": resource_efficiency
    },
    "final_pheromone_levels": paths,
    "final_ratio": paths["short"]["pheromone"] / paths["long"]["pheromone"] if paths["long"]["pheromone"] > 0 else float('inf'),
    "total_path_selections": {
        "short": len(path_selection_history["short"]),
        "long": len(path_selection_history["long"])
    }
}

print(f"\n{'='*60}")
print("           ACO PERFORMANCE ANALYSIS")
print(f"{'='*60}")
print(f"Configuration: {max_iterations} max iterations, target ratio {target_ratio}")
print(f"Convergence time: {convergence_time:.2f} seconds")
print(f"Steps to convergence: {final_step}/{max_iterations}")
print(f"Final pheromone paths: {paths}")
print(f"\nPerformance Metrics:")
print(f"  Exploration Efficiency: {exploration_efficiency:.3f}")
print(f"  Convergence Quality:    {convergence_quality:.3f}")
print(f"  Learning Stability:     {learning_stability:.3f}")
print(f"  Overall Fitness:        {overall_fitness:.3f}")
print(f"\nAdditional Metrics:")
print(f"  Convergence Speed:      {convergence_speed:.3f}")
print(f"  Resource Efficiency:    {resource_efficiency:.3f}")

# Save results (if enabled in config)
if config["performance"]["save_results"]:
    results_dir = config["performance"]["results_directory"]
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{results_dir}/aco_performance_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(performance_results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")

if config["visualization"]["show_plots"]:
    plt.show()

# Plot and save resource usage in one window
plt.figure(figsize=(8,4))
plt.plot(timestamps, cpu_log, label='CPU Usage (%)', marker='o')
plt.plot(timestamps, ram_log, label='RAM Usage (%)', marker='^')
plt.plot(timestamps, gpu_load_log, label='GPU Load (%)', marker='s')
plt.plot(timestamps, gpu_mem_log, label='GPU Mem Usage (%)', marker='x')
plt.xlabel('Time Step')
plt.ylabel('Usage (%)')
plt.title('CPU, RAM, and GPU Usage Over Time (50 Steps)')
plt.legend()
plt.tight_layout()
plt.show()
