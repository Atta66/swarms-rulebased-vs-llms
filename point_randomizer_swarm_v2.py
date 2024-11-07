import pygame
import time
from swarm import Swarm, Agent
import re

class RandomMovingPoints:
    def __init__(self, width=600, height=400, point_size=5, num_points=3):
        # Initialize pygame
        pygame.init()
        self.client = Swarm()
        self.width = width
        self.height = height
        self.point_size = point_size
        self.num_points = num_points  # Number of points to draw
        self.win = pygame.display.set_mode((self.width, self.height))
        self.initial_step = 0
        pygame.display.set_caption("Random Moving Points")

        # Define colors
        self.BLACK = (0, 0, 0)
        self.POINT_COLOR = (255, 0, 0)  # Red points

        # Placeholder for agent1, to be created later
        self.agent1 = None
        self.messages = []

        # Initialize pygame clock for frame rate control
        self.clock = pygame.time.Clock()
        self.running = True

    def create_agent1(self):
        """Create Agent1 with instructions for generating initial coordinates."""
        self.agent1 = Agent(
            name="Agent1",
            instructions=f"""Generate exactly {self.num_points} pairs of (x, y) coordinates in this format: 
            (x1,y1), (x2,y2), (x3,y3). For each pair, x should be between 0 and {self.width - 1}, and y 
            should be between 0 and {self.height - 1}. Provide only the coordinates, with no additional text.""",
        )

    def draw_point(self, x, y):
        """Draw a point at the given coordinates."""
        pygame.draw.circle(self.win, self.POINT_COLOR, (x, y), self.point_size)

    def parse_coordinates(self, coord_string):
        # Use regex to extract numbers with or without surrounding brackets
        coord_pairs = re.findall(r"[\[\(]?(\d+),\s*(\d+)[\]\)]?", coord_string)
        # Convert the pairs from strings to integers
        coords = [(int(x), int(y)) for x, y in coord_pairs]
        return coords

    def run_agent1(self):
        """Run Agent1 once to get the initial coordinates."""
        if self.agent1 is None:
            self.create_agent1()  # Ensure agent1 is created only once

        response = self.client.run(agent=self.agent1, messages=self.messages)
        self.messages = response.messages
        last_message = self.messages[-1]

        if last_message["content"] is not None:
            print("Agent1 response:", last_message["content"])
            coordinates = self.parse_coordinates(last_message["content"])
        else:
            print("Invalid response from Agent1")
        return coordinates

    def run_agent2(self, coordinates):
        """Run Agent2 to increment x values of each coordinate individually."""
        updated_coordinates = []

        for x, y in coordinates:
            coord_string = f"({x},{y})"
            
            agent2 = Agent(
                name="Agent2",
                instructions=f"Increase the x value in {coord_string} by 20 and output the result in the exact same format as the input. Do not add any additional text.",
            )
            
            response = self.client.run(agent=agent2, messages=self.messages)
            last_message = response.messages[-1]

            if last_message["content"] is not None:
                print("Agent2 response:", last_message["content"])
                updated_coordinate = self.parse_coordinates(last_message["content"])
                updated_coordinates.extend(updated_coordinate)  # Add each updated pair to the final list

        return updated_coordinates

    # def run_agent3(self, coordinates):
    #     agent3 = Agent(
    #         name="Agent3",
    #         instructions=f"""Update only the y values in {coordinates} that are greater than 300 by 
    #         setting them to 0. Return the result in the same format as {coordinates}, with no extra 
    #         text and keeping the same number of coordinate pairs as in the input.""",
    #     )
    #     response = self.client.run(agent=agent3, messages=self.messages)
    #     last_message = response.messages[-1]

    #     coordinates = self.run_agent2(coordinates)

    #     return coordinates

    def run(self):
        coordinates = []
        
        while self.running:
            self.win.fill(self.BLACK)
            print(coordinates)

            if self.initial_step == 0:
                # Get response from agent 1
                coordinates = self.run_agent1()
                self.initial_step += 1
            else:
                coordinates = self.run_agent2(coordinates)

            for x, y in coordinates:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.draw_point(x, y)
                    print(f"Point drawn at ({x}, {y})")
                else:
                    print(f"Wrong coordinate values: ({x}, {y}) out of range")

            pygame.display.update()
            time.sleep(1)
            self.messages = []

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.clock.tick(60)

        pygame.quit()

# Create an instance of the game and run it
if __name__ == "__main__":
    game = RandomMovingPoints()
    game.run()
