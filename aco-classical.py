import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import GPUtil

# Setup for resource monitoring
cpu_log = []
ram_log = []
gpu_load_log = []
gpu_mem_log = []
timestamps = []

process = psutil.Process()
start_time = time.time()

evaporation_rate = 0.01

# Two possible paths between A and B
paths = {
    "short": {"distance": 1, "pheromone": 1},
    "long": {"distance": 2, "pheromone": 1}
}

def choose_path(paths):
    total = sum(p["pheromone"] / p["distance"] for p in paths.values())
    probs = {name: (p["pheromone"] / p["distance"]) / total for name, p in paths.items()}
    return np.random.choice(list(probs.keys()), p=list(probs.values()))

# Simulate ants
history = {"short": 0, "long": 0}
for step in range(1000):
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
    paths[chosen]["pheromone"] += 1 / paths[chosen]["distance"]
    for path in paths.values():
        path["pheromone"] *= (1 - evaporation_rate)

    # Plot the pheromone levels
    plt.clf()
    plt.bar(paths.keys(), [p["pheromone"] for p in paths.values()], color=['green', 'blue'])
    plt.title(f"Step {step+1}")
    plt.ylabel("Pheromone Level")
    plt.pause(0.2)


    # Stop if short/long pheromone ratio is close to 236.2
    ratio = paths["short"]["pheromone"] / paths["long"]["pheromone"]
    if abs(ratio - 236.2) < 1.0:
        if step % 100 == 0:
            print(f"Step {step+1}: Short pheromone = {paths['short']['pheromone']:.2f}, Long pheromone = {paths['long']['pheromone']:.2f}, Ratio = {ratio:.2f}")
        print(f"Stopping early: short/long pheromone ratio reached {ratio:.2f} at step {step+1}")
        break

# Final timing and reporting
end_time = time.time()
convergence_time = end_time - start_time
print(f"Convergence time: {convergence_time:.2f} seconds")
print(f"Final pheromone paths: {paths}")
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
