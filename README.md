## Prompts for ACO
### Path Selection Agent Prompt
> Your environment provides two paths with the following characteristics: `paths = {paths}`. Based on this information, choose one of the paths in a way that balances the pheromone level and the distance, following behavior inspired by real ant colonies. Return only the name of the chosen path: `short` or `long`. No additional text.

### Pheromone Update Agent Prompt
> You are a pheromone update agent in an Ant Colony Optimization simulation. The paths are given by `{paths}`. Based on the chosen path which is `{chosen_path}`, update the pheromone levels to reflect the quality of the chosen path and respond with the updated pheromones as an output: `[x, y]` where `x` corresponds to `short` and `y` corresponds to `long`. No additional text.

### Evaporation Agent Prompt
> You are a pheromone evaporation agent. The paths are given by `{paths}`. Based on the evaporation rate `{evaporation_rate}`, apply evaporation to the pheromone levels and return the pheromone levels in the format: `[x, y]` where `x` corresponds to `short` and `y` corresponds to `long`. No additional text.
