import pygame
import json
import os
import math
import random
from swarm import Swarm, Agent

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

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            self.width = self.config["width"]
            self.height = self.config["height"]
            self.point_size = self.config["point_size"]
            self.num_points = self.config.get("num_points", 30)
            self.radius = self.config["radius"]
            self.perception_radius = self.config["perception_radius"]

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
        print(f"Prompt for {agent_name}:\n{prompt}\n")  # Print the prompt
        agent = Agent(name=agent_name, instructions=prompt)
        response = self.client.run(agent=agent, messages=[])
        last_message = response.messages[-1]
        if last_message["content"] is not None:
            # Expecting output in format: (dx, dy)
            try:
                dx, dy = eval(last_message["content"])
                return float(dx), float(dy)
            except Exception:
                print(f"Invalid response from {agent_name}: {last_message['content']}")
        return 0.0, 0.0

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
            sep_prompt = (
                f"You are a boid at position {tuple(pos)}. "
                f"Other boids: {[(tuple(o['position']), tuple(o['velocity'])) for o in others]}. "
                f"Your task is to avoid getting too close to other boids within a radius of {self.radius}. "
                f"Return a (dx, dy) vector representing the separation force to apply to your velocity. "
                f"Only output the vector as (dx, dy). NO additional text."
            )
            coh_prompt = (
                f"You are a boid at position {tuple(pos)}. "
                f"Other boids: {[(tuple(o['position']), tuple(o['velocity'])) for o in others]}. "
                f"Your task is to move slightly toward the average position of nearby boids within a radius of {self.perception_radius}. "
                f"Return a (dx, dy) vector representing the cohesion force to apply to your velocity. "
                f"Only output the vector as (dx, dy). No additional text."
            )
            ali_prompt = (
                f"You are a boid at position {tuple(pos)} with velocity {tuple(vel)}. "
                f"Other boids: {[(tuple(o['position']), tuple(o['velocity'])) for o in others]}. "
                f"Your task is to align your velocity with the average velocity of nearby boids within a radius of {self.perception_radius}. "
                f"Return a (dx, dy) vector representing the alignment force to apply to your velocity. "
                f"Only output the vector as (dx, dy). NO additional text."
            )

            sep_dx, sep_dy = self.run_agent("SeparationAgent", sep_prompt)
            print(f"Separation output: {sep_dx}, {sep_dy}")  # Debugging output
            coh_dx, coh_dy = self.run_agent("CohesionAgent", coh_prompt)
            print(f"Cohesion output: {coh_dx}, {coh_dy}")
            ali_dx, ali_dy = self.run_agent("AlignmentAgent", ali_prompt)
            print(f"Alignment output: {ali_dx}, {ali_dy}")

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
            print(f"new velocities list: {new_velocities}")  # Debugging output

        # Update positions and velocities
        for i in range(self.num_points):
            self.velocities[i] = new_velocities[i]
            self.coordinates[i][0] += self.velocities[i][0]
            self.coordinates[i][1] += self.velocities[i][1]
            # Keep within bounds
            self.coordinates[i][0] = max(0, min(self.width, self.coordinates[i][0]))
            self.coordinates[i][1] = max(0, min(self.height, self.coordinates[i][1]))

    def draw_and_update_points(self):
        self.win.fill(self.BACKGROUND)
        for x, y in self.coordinates:
            self.draw_point(x, y)
        pygame.display.update()

    def run(self):
        import time
        while self.running:
            start_time = time.time()
            self.draw_and_update_points()
            self.update_agents()
            step_time = time.time() - start_time
            print(f"Time for one time step: {step_time:.3f} seconds")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            pygame.time.delay(100)
        pygame.quit()

if __name__ == "__main__":
    game = LLMBoids(config_file="config.json")
    game.run()