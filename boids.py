import pygame
import random
import math

# Constants
WIDTH, HEIGHT = 800, 600
NUM_BOIDS = 100
MAX_SPEED = 4
MAX_FORCE = 0.1
NEIGHBOR_RADIUS = 50

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Boid:
    def __init__(self, x, y):
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * MAX_SPEED
        self.acceleration = pygame.math.Vector2(0, 0)

    def edges(self):
        if self.position.x > WIDTH:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = WIDTH
        
        if self.position.y > HEIGHT:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = HEIGHT

    def align(self, boids):
        steering = pygame.math.Vector2(0, 0)
        total = 0
        avg_vector = pygame.math.Vector2(0, 0)

        for other in boids:
            if other != self and self.position.distance_to(other.position) < NEIGHBOR_RADIUS:
                avg_vector += other.velocity
                total += 1

        if total > 0:
            avg_vector /= total
            avg_vector = avg_vector.normalize() * MAX_SPEED
            steering = avg_vector - self.velocity
            steering = self.limit(steering, MAX_FORCE)

        return steering

    def cohesion(self, boids):
        steering = pygame.math.Vector2(0, 0)
        total = 0
        center_of_mass = pygame.math.Vector2(0, 0)

        for other in boids:
            if other != self and self.position.distance_to(other.position) < NEIGHBOR_RADIUS:
                center_of_mass += other.position
                total += 1

        if total > 0:
            center_of_mass /= total
            steering = center_of_mass - self.position
            steering = steering.normalize() * MAX_SPEED
            steering -= self.velocity
            steering = self.limit(steering, MAX_FORCE)

        return steering

    def separation(self, boids):
        steering = pygame.math.Vector2(0, 0)
        total = 0

        for other in boids:
            distance = self.position.distance_to(other.position)
            if other != self and distance < NEIGHBOR_RADIUS:
                diff = self.position - other.position
                diff /= distance  # Weight by distance
                steering += diff
                total += 1

        if total > 0:
            steering /= total
            steering = steering.normalize() * MAX_SPEED
            steering -= self.velocity
            steering = self.limit(steering, MAX_FORCE)

        return steering

    def limit(self, vector, max_value):
        if vector.length() > max_value:
            return vector.normalize() * max_value
        return vector

    def update(self):
        self.velocity += self.acceleration
        self.velocity = self.limit(self.velocity, MAX_SPEED)
        self.position += self.velocity
        self.acceleration *= 0  # Reset acceleration

    def apply_behaviors(self, boids):
        alignment = self.align(boids)
        cohesion = self.cohesion(boids)
        separation = self.separation(boids)

        self.acceleration += alignment
        self.acceleration += cohesion
        self.acceleration += separation

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Simulation")
    clock = pygame.time.Clock()

    # Create boids
    boids = [Boid(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(NUM_BOIDS)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        for boid in boids:
            boid.edges()
            boid.apply_behaviors(boids)
            boid.update()

            # Draw the boid
            pygame.draw.circle(screen, WHITE, (int(boid.position.x), int(boid.position.y)), 3)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
