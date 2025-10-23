# Swarm Comparison

Comparison of traditional rule-based swarms and LLM-driven swarms using Boids and Ant Colony Optimization.

## Branches

- The `ACO` branch contains the implementation of classical ACO and its LLM-based counterpart.
- The `boids` branch contains the implementation of classical Boids and the LLM-based version.

## Prompts for Boids
### Separation Prompt
> You are a boid at position `(x, y)`. Other boids: `[((x₁, y₁), (vx₁, vy₁)), ...]`. Your task is to avoid getting too close to other boids within a radius of `R`. Return a `(dx, dy)` vector representing the separation force to apply to your velocity. Only output the vector as `(dx, dy)`. No additional text.

### Cohesion Prompt
> You are a boid at position `(x, y)`. Other boids: `[((x₁, y₁), (vx₁, vy₁)), ...]`. Your task is to move slightly toward the average position of nearby boids within a radius of `R`. Return a `(dx, dy)` vector representing the cohesion force to apply to your velocity. Only output the vector as `(dx, dy)`. No additional text.

### Alignment Prompt
> You are a boid at position `(x, y)` with velocity `(vx, vy)`. Other boids: `[((x₁, y₁), (vx₁, vy₁)), ...]`. Your task is to align your velocity with the average velocity of nearby boids within a radius of `R`. Return a `(dx, dy)` vector representing the alignment force to apply to your velocity. Only output the vector as `(dx, dy)`. No additional text.

## Prompts for ACO
### Path Selection Agent Prompt (intial)
> Your environment provides two paths with the following characteristics: `paths = {paths}`. Based on this information, choose one of the paths in a way that balances the pheromone level and the distance, following behavior inspired by real ant colonies. Return only the name of the chosen path: `short` or `long`. No additional text.

### Path Selection Agent Prompt (updated)
> You are an ant in an ACO simulation using an exploration---exploitation strategy. Current step: \{step\} of \{max\_iterations\}. Paths: \{paths\}. Strategy: Early Phase (steps 0---\{early\_phase\_end\}): Explore---choose paths more randomly, aiming for roughly 50/50 exploration. Middle Phase (steps \{mid\_phase\_end\} to \{late\_phase\_start\}): Transition---start considering pheromones but continue occasional exploration. Late Phase (steps \{late\_phase\_start\}): Exploit---focus on the path with the better pheromone-to-distance ratio. Current phase: \{current\_phase\}. In the \{current\_phase\} phase, you should \{phase\_instruction\}. Return only: short or long.

### Pheromone Update Agent Prompt
> You are a pheromone update agent in an Ant Colony Optimization simulation. The paths are given by `{paths}`. Based on the chosen path which is `{chosen_path}`, update the pheromone levels to reflect the quality of the chosen path and respond with the updated pheromones as an output: `[x, y]` where `x` corresponds to `short` and `y` corresponds to `long`. No additional text.

### Evaporation Agent Prompt
> You are a pheromone evaporation agent. The paths are given by `{paths}`. Based on the evaporation rate `{evaporation_rate}`, apply evaporation to the pheromone levels and return the pheromone levels in the format: `[x, y]` where `x` corresponds to `short` and `y` corresponds to `long`. No additional text.

## Results

Detailed comparisons between classical and GPT-based implementations of Boids and ACO are presented in the accompanying paper.  
Here, we provide supplementary results comparing **classical** and **local LLM-based** versions to illustrate relative performance.  
All local experiments were conducted using the **Qwen 2.5 Instruct (14B)** model.

### Boids
<img width="4470" height="2955" alt="classic_vs_llm_comparison_20251009_114508" src="https://github.com/user-attachments/assets/b45e780d-7eae-45bc-981c-30b1bdd682c6" />

In the same configuration and environment, the **local LLM** achieved an overall fitness score of **0.470**, compared to **0.560** for the **GPT-based** implementation. Furthermore, the task was completed in **7468.58 seconds** (about 7x times slower than GPT-based implementation).

### ACO
<img width="3570" height="2955" alt="aco_vs_aco_llm_comparison_20250924_124235" src="https://github.com/user-attachments/assets/2415a639-c69e-4162-8867-6a2139bc511d" />

Under identical conditions, the **local LLM** achieved an overall fitness score of **0.558**, while the **GPT-based** implementation reached **0.862**. The task was completed in **2863.41 seconds** (about 3x times slower than GPT-based implementation).

