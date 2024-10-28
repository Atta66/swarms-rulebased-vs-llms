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
        # self.coordinates = []
        pygame.display.set_caption("Random Moving Points")

        # Define colors
        self.BLACK = (0, 0, 0)
        self.POINT_COLOR = (255, 0, 0)  # Red points

        # Set up agent with instructions for 3 points
        self.agent1 = Agent(
            name="Agent1",
            instructions=f"""respond with {self.num_points} pairs of x and y coordinates.
            x should be between 0 and {self.width - 1} and y should be between 0 and {self.height - 1}.
            Just write the coordinates as space-separated pairs, do not write anything other than the coordinates.
            do not use brackets and commas, the format should be exactly in the format: (x1,y1), (x2,y2), (x3,y3) and
            no additional text.""",
        )


        self.messages = []

        # Initialize pygame clock for frame rate control
        self.clock = pygame.time.Clock()
        self.running = True

    def draw_point(self, x, y):
        """Draw a point at the given coordinates."""
        pygame.draw.circle(self.win, self.POINT_COLOR, (x, y), self.point_size)

    # def parse_coordinates(self, coord_string):
    #     # Use regex to extract all numbers inside parentheses
    #     print("coord_string",coord_string)
    #     coord_pairs = re.findall(r"\((\d+),(\d+)\)", coord_string)
    #     print("coord_pairs", coord_pairs)
    #     # Convert the pairs from strings to integers
    #     coords = [(int(x), int(y)) for x, y in coord_pairs]
    #     return coords

    def parse_coordinates(self, coord_string):
        # Use regex to extract numbers with or without surrounding brackets
        print("coord_string", coord_string)
        coord_pairs = re.findall(r"[\[\(]?(\d+),\s*(\d+)[\]\)]?", coord_string)
        print("coord_pairs", coord_pairs)
        # Convert the pairs from strings to integers
        coords = [(int(x), int(y)) for x, y in coord_pairs]
        return coords


    def run_agent1(self):
        """Run Agent1 once to get the initial coordinates."""
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
        """Run Agent2 in the loop to increment the coordinates."""
        # coord_string = ', '.join([f"({x},{y})" for x, y in self.coordinates])
        print("i am here", coordinates)
        agent2 = Agent(
            name="Agent2",
            instructions=f"""increment the x in {coordinates} by 20 and output in the same format as the input, there should not be any additional text""",
        )
        response = self.client.run(agent=agent2, messages=self.messages)
        last_message = response.messages[-1]
        print("last_message:", last_message)

        if last_message["content"] is not None:
            print("Agent2 response:", last_message["content"])
            coordinates = self.parse_coordinates(last_message["content"])
            print(coordinates)    
        return coordinates

    def run_agent3(self, coordinates):

        agent3 = Agent(
            name="Agent3",
            instructions=f"""if any value in {coordinates} is greater than 300, make it 0 and output the updated list in the same format as the input, no additional text""",
        )
        response = self.client.run(agent=agent3, messages=self.messages)
        last_message = response.messages[-1]
        coordinates = self.parse_coordinates(last_message["content"])

        coordinates = self.run_agent2(coordinates)

        return coordinates



    def run(self):

        coordinates = []
        
        while self.running:
            # Fill the background
            self.win.fill(self.BLACK)
            print(coordinates)

            if self.initial_step == 0:
                # Get response from agent 1
                coordinates = self.run_agent1()
                self.initial_step += 1
            else:
                # Get response from agent 2
                # print(self.coordinates, "i am here")
                coordinates = self.run_agent3(coordinates)

            # self.coordinates = self.coordinates
                    # Draw each point or skip if out of range
            for x, y in coordinates:
                # Check if coordinates are out of range
                if 0 <= x < 600 and 0 <= y < 400:
                    self.draw_point(x, y)
                    print(f"Point drawn at ({x}, {y})")
                else:
                    print(f"Wrong coordinate values: ({x}, {y}) out of range")

            # Update the display
            pygame.display.update()

            # Wait for 1 second before generating a new position
            time.sleep(1)

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
    game = RandomMovingPoints()
    game.run()
