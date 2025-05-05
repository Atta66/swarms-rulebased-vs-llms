import numpy as np
import matplotlib.pyplot as plt

# Two possible paths between A and B
paths = {
    "short": {"distance": 1, "pheromone": 1},
    "long": {"distance": 3, "pheromone": 1}
}

def choose_path(paths):
    # Probability ∝ pheromone / distance
    total = sum(p["pheromone"] / p["distance"] for p in paths.values())
    print(f"Total: {total}")
    probs = {name: (p["pheromone"] / p["distance"]) / total for name, p in paths.items()}
    print(f"Probabilities: {probs}")
    return np.random.choice(list(probs.keys()), p=list(probs.values()))

# Simulate ants
history = {"short": 0, "long": 0}
for step in range(50):
    chosen = choose_path(paths)
    history[chosen] += 1
    # reinforce the chosen path
    paths[chosen]["pheromone"] += 1

    # Plot the pheromone levels
    plt.clf()
    plt.bar(paths.keys(), [p["pheromone"] for p in paths.values()], color=['green', 'blue'])
    plt.title(f"Step {step+1}")
    plt.ylabel("Pheromone Level")
    plt.pause(0.2)

plt.show()
