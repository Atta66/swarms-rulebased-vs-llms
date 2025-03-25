import pygame
import json
from swarm import Swarm, Agent
import os
import math

class RandomMovingPoints:
    def __init__(self, config_file="config.json"):
        base_path = "C:\\Users\\attah\\OneDrive\\Desktop\\workspace\\ACES\\swarm_openai\\ACES"
        config_path = os.path.join(base_path, "config", config_file)

        self.load_config(config_path)
        
        # Initialize pygame
        pygame.init()
        self.client = Swarm()
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Attraction and Repulsion Simulation")

        # Define colors
        self.BACKGROUND = tuple(self.config["colors"]["WHITE"])
        self.POINT_COLOR = tuple(self.config["colors"]["BLUE"])

        self.messages = []
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
                self.agent_config = self.config["agent1"]
                self.coordinates = self.config["coordinates"]
        
        except FileNotFoundError:
            print(f"Config file {config_file} not found.")
            raise
        except json.JSONDecodeError:
            print(f"Error decoding the config file {config_file}.")
            raise
 
    def draw_point(self, x, y):
        """Draw a point at the given coordinates."""
        pygame.draw.circle(self.win, self.POINT_COLOR, (int(x), int(y)), self.point_size)

    def run_agent(self, agent_config, position, other_positions):
        """Run an agent to calculate its new position."""
        instructions = agent_config["instructions"]
        instructions = instructions.format(
            position=position,
            other_positions=other_positions,
            radius=self.radius,
            width=self.width,
            height=self.height
        )

        agent = Agent(
            name=agent_config["name"],
            instructions=instructions
        )

        response = self.client.run(agent=agent, messages=self.messages)
        last_message = response.messages[-1]

        if last_message["content"] is not None:
            new_position = eval(last_message["content"])  # Parse the returned position
        else:
            print(f"Invalid response from {agent_config['name']}")
            new_position = position  # Keep the old position if the response is invalid

        return new_position

    def update_positions(self):
        """Update the positions of all particles."""
        new_coordinates = []
        for i in range(self.num_points):
            position = self.coordinates[i]
            other_positions = [self.coordinates[j] for j in range(self.num_points) if j != i]

            # Run the agent to calculate its new position
            new_position = self.run_agent(
                self.agent_config,
                position=position,
                other_positions=other_positions
            )
            new_coordinates.append(new_position)

        self.coordinates = new_coordinates
        print("New coordinates: ", self.coordinates)

    def draw_and_update_points(self):
        """Draw each point on the screen and update their positions."""
        self.win.fill(self.BACKGROUND)

        # Draw all particles
        for x, y in self.coordinates:
            self.draw_point(x, y)

        pygame.display.update()

    def run(self):
        while self.running:
            self.draw_and_update_points()
            self.update_positions()
            self.draw_and_update_points()
            self.messages = []

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            pygame.time.delay(100)

        pygame.quit()

# Create an instance of the game and run it
if __name__ == "__main__":
    game = RandomMovingPoints(config_file="config.json")
    game.run()
