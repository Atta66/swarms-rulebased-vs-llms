import pygame
import time
import json
from swarm import Swarm, Agent
import re
import math
import os

class RandomMovingPoints:
    def __init__(self, config_file="config.json"):

        base_path = "C:\\Users\\Atta\\Desktop\\workspace\\ACES\\swarm_ai\\ACES"
        # base_path = "C:\\Users\\attah\\OneDrive\\Desktop\\workspace\\ACES\\swarm_openai\\ACES"
        config_path = os.path.join(base_path, "config", config_file)

        self.load_config(config_path)
        
        # Initialize pygame
        pygame.init()
        self.client = Swarm()
        self.win = pygame.display.set_mode((self.width, self.height))
        self.initial_step = 0
        pygame.display.set_caption("Random Moving Points")

        # Define colors
        self.BACKGROUND = tuple(self.config["colors"]["WHITE"])
        self.POINT_COLOR = tuple(self.config["colors"]["BLUE"])

        self.messages = []

        # Initialize pygame clock for frame rate control
        self.clock = pygame.time.Clock()
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
                self.agent1_config = self.config["agent1"]
                self.agent2_config = self.config["agent2"]
                self.agent3_config = self.config["agent3"]
                self.agent4_config = self.config["agent4"]
        
        except FileNotFoundError:
            print(f"Config file {config_file} not found.")
            raise
        except json.JSONDecodeError:
            print(f"Error decoding the config file {config_file}.")
            raise

    def draw_point(self, x, y):
        """Draw a point at the given coordinates."""
        pygame.draw.circle(self.win, self.POINT_COLOR, (x, y), self.point_size)

    def parse_coordinates(self, coord_string):
        # Use regex to extract numbers with or without surrounding brackets
        coord_pairs = re.findall(r"[\[\(]?(\d+),\s*(\d+)[\]\)]?", coord_string)
        # Convert the pairs from strings to integers
        coords = [(int(x), int(y)) for x, y in coord_pairs]
        return coords

    def save_coordinates_to_config(self, coordinates):
        """Update and save coordinates back to the config file."""
        self.config["coordinates"] = coordinates  # Update the 'coordinates' key in the config

        base_path = "C:\\Users\\Atta\\Desktop\\workspace\\ACES\\swarm_ai\\ACES"
        # base_path = "C:\\Users\\attah\\OneDrive\\Desktop\\workspace\\ACES\\swarm_openai\\ACES"
        config_path = os.path.join(base_path, "config", "config.json")

        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)  # Write the updated config back to the file

    def run_agent1(self):
        """Run Agent1 once to get the initial coordinates."""

        # Get the instructions and replace placeholders with the actual values
        instructions = self.agent1_config["instructions"]
        instructions = instructions.format(num_points=self.num_points, width=self.width, height=self.height)

        agent1 = Agent(
            name=self.agent1_config["name"],
            instructions=instructions
        )

        response = self.client.run(agent=agent1, messages=self.messages)
        # self.messages = response.messages
        last_message = response.messages[-1]

        if last_message["content"] is not None:
            coordinates = self.parse_coordinates(last_message["content"])
        else:
            print("Invalid response from Agent1")
        print("generated coordinates (agent1): ", coordinates)

        self.save_coordinates_to_config(coordinates)
        return coordinates

    def run_agent2(self, coordinates):
        """Run Agent2 in the loop to increment the coordinates."""

        # Get the instructions and replace placeholders with the actual values
        instructions = self.agent2_config["instructions"]
        instructions = instructions.format(coordinates=coordinates)

        agent2 = Agent(
            name=self.agent2_config["name"],
            instructions=instructions
        )

        response = self.client.run(agent=agent2, messages=self.messages)
        last_message = response.messages[-1]

        if last_message["content"] is not None:
            coordinates = self.parse_coordinates(last_message["content"])
        print("incremented coordinates (agent2): ", coordinates)
        return coordinates

    def run_agent3(self, coordinates):
        """Update y values greater than 300 to 0."""

        instructions = self.agent3_config["instructions"]
        instructions = instructions.format(coordinates=coordinates)

        agent3 = Agent(
            name=self.agent3_config["name"],
            instructions=instructions
        )

        response = self.client.run(agent=agent3, messages=self.messages)
        last_message = response.messages[-1]
        coordinates = self.parse_coordinates(last_message["content"])
        print("boundary check coordinates (agent3): ", coordinates)
        return coordinates

    def run_agent4(self, coordinates):
        """Use Agent4 to check each point's radius and move in the -x direction if another point is inside its radius."""
        # Create the agent with the appropriate instructions for checking distances
        
        instructions = self.agent4_config["instructions"]
        instructions = instructions.format(coordinates=coordinates, radius=self.radius)

        agent4 = Agent(
            name=self.agent4_config["name"],
            instructions=instructions
        )
        
        # Call the agent with the given instructions
        response = self.client.run(agent=agent4, messages=self.messages)
        last_message = response.messages[-1]
        
        # Parse the returned coordinates
        if last_message["content"] is not None:
            coordinates = self.parse_coordinates(last_message["content"])
            print("adjusted coordinates (agent4): ", coordinates)
        else:
            print("Invalid response from Agent4")
        
        return coordinates

    def draw_and_update_points(self, coordinates):
        """Draw each point on the screen if it is within range, update display, and pause."""
        # Clear the background
        self.win.fill(self.BACKGROUND)
        
        # Draw each point or skip if out of range
        for x, y in coordinates:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.draw_point(x, y)

        # Update the display
        pygame.display.update()

        # Wait for 1 second before generating a new position
        # time.sleep()

    def run(self):
        coordinates = []
        
        while self.running:
            # Fill the background
            self.win.fill(self.BACKGROUND)

            if self.initial_step == 0:
                # Get response from agent 1
                coordinates = self.run_agent1()
                self.initial_step += 1
                self.draw_and_update_points(coordinates)
            else:
                # Get response from agent 3 (including agent 2 steps)
                coordinates = self.run_agent2(coordinates)
                self.draw_and_update_points(coordinates)

                coordinates = self.run_agent3(coordinates)
                self.draw_and_update_points(coordinates)

                # Run Agent4 to check if points are within radius of each other
                coordinates = self.run_agent4(coordinates)
                self.draw_and_update_points(coordinates)
                # break

            # Reset messages for the next iteration
            self.messages = []

        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        # Control the frame rate
        self.clock.tick(60)

        # Quit pygame when the loop finishes
        pygame.quit()


# Create an instance of the game and run it
if __name__ == "__main__":
    game = RandomMovingPoints(config_file="config.json")
    game.run()
