import pygame
import json
import os
import math
import random
import time
import psutil
import GPUtil
import matplotlib.pyplot as plt
from agent_performance_tracker import AgentPerformanceTracker

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
        
        # Initialize performance tracker for classic boids
        self.performance_tracker = AgentPerformanceTracker(config_path)
        # Use same expected ranges as LLM boids for fair comparison
        self.performance_tracker.expected_ranges.update({
            "ClassicSeparation": (-5.0, 5.0),   # Same as LLM SeparationAgent
            "ClassicCohesion": (-3.0, 3.0),     # Same as LLM CohesionAgent  
            "ClassicAlignment": (-4.0, 4.0)     # Same as LLM AlignmentAgent
        })

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
                self.simulation_steps = self.config.get("simulation_steps", 10)
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
        start_time = time.time()
        
        steer = [0.0, 0.0]
        count = 0
        for j, pos in enumerate(self.coordinates):
            if i != j:
                dist = math.hypot(self.coordinates[i][0] - pos[0], self.coordinates[i][1] - pos[1])
                if dist < self.radius and dist > 0:
                    # Calculate direction vector (away from neighbor)
                    diff_x = self.coordinates[i][0] - pos[0]
                    diff_y = self.coordinates[i][1] - pos[1]
                    # Normalize by distance to get stronger force when closer
                    steer[0] += diff_x / dist
                    steer[1] += diff_y / dist
                    count += 1
        
        if count > 0:
            # Average and normalize to reasonable range similar to LLM output
            steer[0] = (steer[0] / count) * 2.0  # Scale to match LLM range
            steer[1] = (steer[1] / count) * 2.0
        
        # Track performance
        end_time = time.time()
        self.performance_tracker.track_agent_call("ClassicSeparation", start_time, end_time, tuple(steer))
        
        return steer

    def cohesion(self, i):
        """Move toward the average position of nearby boids."""
        start_time = time.time()
        
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
            # Calculate direction to center and normalize to LLM-like range
            direction_x = center[0] - self.coordinates[i][0]
            direction_y = center[1] - self.coordinates[i][1]
            distance_to_center = math.hypot(direction_x, direction_y)
            
            if distance_to_center > 0:
                # Normalize and scale to match LLM output range
                result = [
                    (direction_x / distance_to_center) * min(2.0, distance_to_center / 50.0),
                    (direction_y / distance_to_center) * min(2.0, distance_to_center / 50.0)
                ]
            else:
                result = [0.0, 0.0]
        else:
            result = [0.0, 0.0]
        
        # Track performance
        end_time = time.time()
        self.performance_tracker.track_agent_call("ClassicCohesion", start_time, end_time, tuple(result))
        
        return result

    def alignment(self, i):
        """Align velocity with the average velocity of nearby boids."""
        start_time = time.time()
        
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
            # Calculate velocity difference and scale to match LLM range
            vel_diff_x = avg_vel[0] - self.velocities[i][0]
            vel_diff_y = avg_vel[1] - self.velocities[i][1]
            
            # Scale to reasonable alignment force (similar to LLM output)
            result = [vel_diff_x * 0.8, vel_diff_y * 0.8]
        else:
            result = [0.0, 0.0]
        
        # Track performance
        end_time = time.time()
        self.performance_tracker.track_agent_call("ClassicAlignment", start_time, end_time, tuple(result))
        
        return result

    def update_agents(self):
        """Update the positions and velocities of all boids using separation, cohesion, and alignment."""
        # Use same weights as LLM boids for fair comparison
        sep_weight = 1.5
        coh_weight = 0.5
        ali_weight = 1.0
        
        new_velocities = []
        for i in range(self.num_points):
            sep = self.separation(i)
            coh = self.cohesion(i)
            ali = self.alignment(i)

            # Combine the three rules with same weights as LLM boids
            self.velocities[i][0] += sep_weight * sep[0] + coh_weight * coh[0] + ali_weight * ali[0]
            self.velocities[i][1] += sep_weight * sep[1] + coh_weight * coh[1] + ali_weight * ali[1]
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
        
        # Evaluate swarm performance after position updates
        self.performance_tracker.evaluate_swarm_performance(
            self.coordinates, self.velocities, self.radius, self.perception_radius
        )

    def draw_and_update_points(self):
        """Draw each point on the screen and update their positions."""
        self.win.fill(self.BACKGROUND)

        # Draw all boids
        for x, y in self.coordinates:
            self.draw_point(x, y)

        pygame.display.update()

    def run(self):
        steps = self.simulation_steps
        total_time = 0.0
        cpu_usages = []
        ram_usages = []
        gpu_loads = []
        gpu_mem_usages = []
        for _ in range(steps):
            start_time = time.time()
            self.draw_and_update_points()
            self.update_agents()
            step_time = time.time() - start_time
            total_time += step_time
            # System resource usage
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().percent
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_load = gpu.load * 100
                gpu_mem_usage = gpu.memoryUtil * 100
            else:
                gpu_load = 0
                gpu_mem_usage = 0
            cpu_usages.append(cpu_usage)
            ram_usages.append(ram_usage)
            gpu_loads.append(gpu_load)
            gpu_mem_usages.append(gpu_mem_usage)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            pygame.time.delay(100)
        # Plot CPU, RAM, GPU usage over time
        plt.figure(figsize=(8,4))
        plt.plot(range(steps), cpu_usages, label='CPU Usage (%)', marker='o')
        plt.plot(range(steps), ram_usages, label='RAM Usage (%)', marker='s')
        plt.plot(range(steps), gpu_loads, label='GPU Load (%)', marker='^')
        plt.plot(range(steps), gpu_mem_usages, label='GPU Mem Usage (%)', marker='x')
        plt.xlabel('Time Step')
        plt.ylabel('Usage (%)')
        plt.title(f'CPU, RAM, GPU Usage Over Time ({steps} Steps)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Print performance summary at the end
        self.performance_tracker.print_performance_summary()
        
        # Save performance data
        self.performance_tracker.save_performance_data("classic_boids_performance.json")
        
        print(f"Total time for {steps} time steps: {total_time:.10f} seconds")
        print(f"Average CPU Usage: {sum(cpu_usages)/len(cpu_usages):.2f}%")
        print(f"Average RAM Usage: {sum(ram_usages)/len(ram_usages):.2f}%")
        print(f"Average GPU Load: {sum(gpu_loads)/len(gpu_loads):.2f}%")
        print(f"Average GPU Memory Usage: {sum(gpu_mem_usages)/len(gpu_mem_usages):.2f}%")
        pygame.quit()

# Create an instance of the simulation and run it
if __name__ == "__main__":
    game = ClassicBoids(config_file="config.json")
    game.run()