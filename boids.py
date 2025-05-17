import pygame
import json
import os
import math
import random

class ClassicBoids:
    def __init__(self, config_file="config.json"):
        # Adjust the path accordingly
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, "config", config_file)

        self.load_config(config_path)

        # Initialize pygame
        pygame.init()
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Classic Boids Simulation")

        # Define colors
        self.BACKGROUND = tuple(self.config["colors"]["WHITE"])
        self.POINT_COLOR = tuple(self.config["colors"]["BLUE"])

        self.running = True

    def load_config(self, config_file):
        """Load configuration from the JSON file."""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)

                self.width = self.config["width"]
                self.height = self.config["height"]
                self.point_size = self.config["point_size"]
                self.num_points = self.config["num_points"]
                self.radius = self.config["radius"]
                self.perception_radius = self.config["perception_radius"]
                # Initialize coordinates and velocities
                coords = self.config.get("coordinates", [])
                vels = self.config.get("velocities", [])
                while len(coords) < self.num_points:
                    coords.append([
                        random.uniform(0, self.width),
                        random.uniform(0, self.height)
                    ])
                while len(vels) < self.num_points:
                    vels.append([
                        random.uniform(-2, 2),
                        random.uniform(-2, 2)
                    ])
                self.coordinates = coords
                self.velocities = vels
        except FileNotFoundError:
            print(f"Config file {config_file} not found.")
            raise
        except json.JSONDecodeError:
            print(f"Error decoding the config file {config_file}.")
            raise

    def draw_point(self, x, y):
        """Draw a point at the given coordinates."""
        pygame.draw.circle(self.win, self.POINT_COLOR, (int(x), int(y)), self.point_size)

    def limit_velocity(self, velocity, max_speed=4):
        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        if speed > max_speed:
            velocity[0] = (velocity[0] / speed) * max_speed
            velocity[1] = (velocity[1] / speed) * max_speed
        return velocity

    def separation(self, i):
        """Move away from nearby boids to avoid crowding."""
        steer = [0.0, 0.0]
        count = 0
        for j, pos in enumerate(self.coordinates):
            if i != j:
                dist = math.hypot(self.coordinates[i][0] - pos[0], self.coordinates[i][1] - pos[1])
                if dist < self.radius and dist > 0:
                    steer[0] += self.coordinates[i][0] - pos[0]
                    steer[1] += self.coordinates[i][1] - pos[1]
                    count += 1
        if count > 0:
            steer[0] /= count
            steer[1] /= count
        return steer

    def cohesion(self, i):
        """Move toward the average position of nearby boids."""
        center = [0.0, 0.0]
        count = 0
        for j, pos in enumerate(self.coordinates):
            if i != j:
                dist = math.hypot(self.coordinates[i][0] - pos[0], self.coordinates[i][1] - pos[1])
                if dist < self.perception_radius:
                    center[0] += pos[0]
                    center[1] += pos[1]
                    count += 1
        if count > 0:
            center[0] /= count
            center[1] /= count
            return [(center[0] - self.coordinates[i][0]) * 0.01, (center[1] - self.coordinates[i][1]) * 0.01]
        return [0.0, 0.0]

    def alignment(self, i):
        """Align velocity with the average velocity of nearby boids."""
        avg_vel = [0.0, 0.0]
        count = 0
        for j, pos in enumerate(self.coordinates):
            if i != j:
                dist = math.hypot(self.coordinates[i][0] - pos[0], self.coordinates[i][1] - pos[1])
                if dist < self.perception_radius:
                    avg_vel[0] += self.velocities[j][0]
                    avg_vel[1] += self.velocities[j][1]
                    count += 1
        if count > 0:
            avg_vel[0] /= count
            avg_vel[1] /= count
            return [(avg_vel[0] - self.velocities[i][0]) * 0.05, (avg_vel[1] - self.velocities[i][1]) * 0.05]
        return [0.0, 0.0]

    def update_agents(self):
        """Update the positions and velocities of all boids using separation, cohesion, and alignment."""
        new_velocities = []
        for i in range(self.num_points):
            sep = self.separation(i)
            coh = self.cohesion(i)
            ali = self.alignment(i)

            # Combine the three rules
            self.velocities[i][0] += sep[0] + coh[0] + ali[0]
            self.velocities[i][1] += sep[1] + coh[1] + ali[1]
            self.velocities[i] = self.limit_velocity(self.velocities[i])

            new_velocities.append(list(self.velocities[i]))

        # Update positions
        for i in range(self.num_points):
            self.coordinates[i][0] += self.velocities[i][0]
            self.coordinates[i][1] += self.velocities[i][1]

            # Keep boids within the window boundaries
            self.coordinates[i][0] = max(0, min(self.width, self.coordinates[i][0]))
            self.coordinates[i][1] = max(0, min(self.height, self.coordinates[i][1]))

        self.velocities = new_velocities

    def draw_and_update_points(self):
        """Draw each point on the screen and update their positions."""
        self.win.fill(self.BACKGROUND)

        # Draw all boids
        for x, y in self.coordinates:
            self.draw_point(x, y)

        pygame.display.update()

    def run(self):
        import time
        steps = 10
        total_time = 0.0
        for _ in range(steps):
            start_time = time.time()
            self.draw_and_update_points()
            self.update_agents()
            step_time = time.time() - start_time
            total_time += step_time
            print(f"Time for one time step: {step_time:.10f} seconds")
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            pygame.time.delay(100)
        print(f"Total time for {steps} time steps: {total_time:.10f} seconds")
        pygame.quit()

# Create an instance of the simulation and run it
if __name__ == "__main__":
    game = ClassicBoids(config_file="config.json")
    game.run()