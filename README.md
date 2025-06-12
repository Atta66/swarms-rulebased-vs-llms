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
### Path Selection Agent Prompt
> Your environment provides two paths with the following characteristics: `paths = {paths}`. Based on this information, choose one of the paths in a way that balances the pheromone level and the distance, following behavior inspired by real ant colonies. Return only the name of the chosen path: `short` or `long`. No additional text.

### Pheromone Update Agent Prompt
> You are a pheromone update agent in an Ant Colony Optimization simulation. The paths are given by `{paths}`. Based on the chosen path which is `{chosen_path}`, update the pheromone levels to reflect the quality of the chosen path and respond with the updated pheromones as an output: `[x, y]` where `x` corresponds to `short` and `y` corresponds to `long`. No additional text.

### Evaporation Agent Prompt
> You are a pheromone evaporation agent. The paths are given by `{paths}`. Based on the evaporation rate `{evaporation_rate}`, apply evaporation to the pheromone levels and return the pheromone levels in the format: `[x, y]` where `x` corresponds to `short` and `y` corresponds to `long`. No additional text.
