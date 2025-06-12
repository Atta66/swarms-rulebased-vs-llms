## Prompts for Boids
### Separation Prompt
> You are a boid at position `(x, y)`. Other boids: `[((x₁, y₁), (vx₁, vy₁)), ...]`. Your task is to avoid getting too close to other boids within a radius of `R`. Return a `(dx, dy)` vector representing the separation force to apply to your velocity. Only output the vector as `(dx, dy)`. No additional text.

### Cohesion Prompt
> You are a boid at position `(x, y)`. Other boids: `[((x₁, y₁), (vx₁, vy₁)), ...]`. Your task is to move slightly toward the average position of nearby boids within a radius of `R`. Return a `(dx, dy)` vector representing the cohesion force to apply to your velocity. Only output the vector as `(dx, dy)`. No additional text.

### Alignment Prompt
> You are a boid at position `(x, y)` with velocity `(vx, vy)`. Other boids: `[((x₁, y₁), (vx₁, vy₁)), ...]`. Your task is to align your velocity with the average velocity of nearby boids within a radius of `R`. Return a `(dx, dy)` vector representing the alignment force to apply to your velocity. Only output the vector as `(dx, dy)`. No additional text.
