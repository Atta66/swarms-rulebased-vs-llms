import pygame
import json
import os
import math
import random
import time
from swarm import Swarm, Agent
import psutil
import GPUtil
import matplotlib.pyplot as plt
from agent_performance_tracker import AgentPerformanceTracker

class LLMBoids:
    def __init__(self, config_file="config.json"):
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, "config", config_file)
        self.load_config(config_path)

        pygame.init()
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("LLM Boids Simulation")

        self.BACKGROUND = tuple(self.config["colors"]["WHITE"])
        self.POINT_COLOR = tuple(self.config["colors"]["BLUE"])
        self.running = True
        self.client = Swarm()
        
        # Initialize performance tracker
        self.performance_tracker = AgentPerformanceTracker(config_path)

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            self.width = self.config["width"]
            self.height = self.config["height"]
            self.point_size = self.config["point_size"]
            self.num_points = self.config.get("num_points", 30)
            self.radius = self.config["radius"]
            self.perception_radius = self.config["perception_radius"]
            self.simulation_steps = self.config.get("simulation_steps", 10)
            self.prompts = self.config.get("prompts", {})

        # Use coordinates and velocities from config if present, else generate random
        self.coordinates = []
        self.velocities = []
        config_coords = self.config.get("coordinates", [])
        config_vels = self.config.get("velocities", [])
        for i in range(self.num_points):
            if i < len(config_coords):
                x, y = config_coords[i]
            else:
                x = random.uniform(0, self.width)
                y = random.uniform(0, self.height)
            self.coordinates.append([x, y])

            if i < len(config_vels):
                vx, vy = config_vels[i]
            else:
                vx = random.uniform(-2, 2)
                vy = random.uniform(-2, 2)
            self.velocities.append([vx, vy])

    def draw_point(self, x, y):
        pygame.draw.circle(self.win, self.POINT_COLOR, (int(x), int(y)), self.point_size)

    def run_agent(self, agent_name, prompt):
        # Record start time for performance tracking
        start_time = time.time()
        
        agent = Agent(name=agent_name, instructions=prompt)
        response = self.client.run(agent=agent, messages=[])
        last_message = response.messages[-1]
        
        # Default return values
        dx, dy = 0.0, 0.0
        
        if last_message["content"] is not None:
            # Expecting output in format: (dx, dy)
            try:
                dx, dy = eval(last_message["content"])
                dx, dy = float(dx), float(dy)
            except Exception as e:
                print(f"Invalid response from {agent_name}: {last_message['content']} - Error: {e}")
        
        # Record end time and track performance
        end_time = time.time()
        self.performance_tracker.track_agent_call(agent_name, start_time, end_time, (dx, dy))
        
        return dx, dy

    def get_other_boids(self, i):
        others = []
        for j, (pos, vel) in enumerate(zip(self.coordinates, self.velocities)):
            if i != j:
                others.append({"position": pos, "velocity": vel})
        return others

    def update_agents(self):
        sep_weight = 1.5
        coh_weight = 0.5
        ali_weight = 1.0
        new_velocities = []

        for i in range(self.num_points):
            pos = self.coordinates[i]
            vel = self.velocities[i]
            others = self.get_other_boids(i)

            # Prompts for each rule
            sep_prompt = self.prompts.get("separation", "").format(
                position=tuple(pos),
                other_boids=[(tuple(o['position']), tuple(o['velocity'])) for o in others],
                radius=self.radius
            )
            coh_prompt = self.prompts.get("cohesion", "").format(
                position=tuple(pos),
                other_boids=[(tuple(o['position']), tuple(o['velocity'])) for o in others],
                perception_radius=self.perception_radius
            )
            ali_prompt = self.prompts.get("alignment", "").format(
                position=tuple(pos),
                velocity=tuple(vel),
                other_boids=[(tuple(o['position']), tuple(o['velocity'])) for o in others],
                perception_radius=self.perception_radius
            )

            sep_dx, sep_dy = self.run_agent("SeparationAgent", sep_prompt)
            coh_dx, coh_dy = self.run_agent("CohesionAgent", coh_prompt)
            ali_dx, ali_dy = self.run_agent("AlignmentAgent", ali_prompt)

            # Combine the three rules
            vx = vel[0] + sep_weight * sep_dx + coh_weight * coh_dx + ali_weight * ali_dx
            vy = vel[1] + sep_weight * sep_dy + coh_weight * coh_dy + ali_weight * ali_dy

            # Limit velocity
            speed = math.sqrt(vx ** 2 + vy ** 2)
            max_speed = 4
            if speed > max_speed:
                vx = (vx / speed) * max_speed
                vy = (vy / speed) * max_speed

            new_velocities.append([vx, vy])

        # Update positions and velocities
        for i in range(self.num_points):
            self.velocities[i] = new_velocities[i]
            self.coordinates[i][0] += self.velocities[i][0]
            self.coordinates[i][1] += self.velocities[i][1]
            # Keep within bounds
            self.coordinates[i][0] = max(0, min(self.width, self.coordinates[i][0]))
            self.coordinates[i][1] = max(0, min(self.height, self.coordinates[i][1]))
        
        # Evaluate swarm performance after position updates
        self.performance_tracker.evaluate_swarm_performance(
            self.coordinates, self.velocities, self.radius, self.perception_radius
        )

    def draw_and_update_points(self):
        self.win.fill(self.BACKGROUND)
        for x, y in self.coordinates:
            self.draw_point(x, y)
        pygame.display.update()

    def run(self):
        cpu_usages = []
        ram_usages = []
        gpu_loads = []
        gpu_mem_usages = []
        time_steps = []
        step_count = 0
        while self.running:
            start_time = time.time()
            self.draw_and_update_points()
            self.update_agents()
            step_time = time.time() - start_time
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
            time_steps.append(step_count)
            step_count += 1
            if step_count >= self.simulation_steps:
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            pygame.time.delay(100)
        
        # Print performance summary at the end
        self.performance_tracker.print_performance_summary()
        
        # Save performance data
        self.performance_tracker.save_performance_data("performance_data.json")
        
        plt.figure(figsize=(8,4))
        plt.plot(time_steps, cpu_usages, label='CPU Usage (%)', marker='o')
        plt.plot(time_steps, ram_usages, label='RAM Usage (%)', marker='^')
        plt.plot(time_steps, gpu_loads, label='GPU Load (%)', marker='s')
        plt.plot(time_steps, gpu_mem_usages, label='GPU Mem Usage (%)', marker='x')
        plt.xlabel('Time Step')
        plt.ylabel('Usage (%)')
        plt.title(f'CPU, RAM, and GPU Usage Over Time ({self.simulation_steps} Steps)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        pygame.quit()

if __name__ == "__main__":
    game = LLMBoids(config_file="config.json")
    game.run()