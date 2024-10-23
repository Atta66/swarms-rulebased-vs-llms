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
        pygame.display.set_caption("Random Moving Points")

        # Define colors
        self.BLACK = (0, 0, 0)
        self.POINT_COLOR = (255, 0, 0)  # Red points

        # Set up agent with instructions for 3 points
        self.agent = Agent(
            name="Agent",
            instructions=f"""respond with {self.num_points} pairs of x and y coordinates.
            x should be between 0 and {self.width - 1} and y should be between 0 and {self.height - 1}.
            Just write the coordinates as space-separated pairs, do not write anything other than the coordinates.
            do not use brackets and commas, the format should be exactly in the format (x1,y1), (x2,y2), (x3,y3) and
            no additional text""",
        )
        self.messages = []

        # Initialize pygame clock for frame rate control
        self.clock = pygame.time.Clock()
        self.running = True

    def draw_point(self, x, y):
        """Draw a point at the given coordinates."""
        pygame.draw.circle(self.win, self.POINT_COLOR, (x, y), self.point_size)

    def parse_coordinates(self, coord_string):
        # Use regex to extract all numbers inside parentheses
        coord_pairs = re.findall(r"\((\d+),(\d+)\)", coord_string)
        # Convert the pairs from strings to integers
        coords = [(int(x), int(y)) for x, y in coord_pairs]
        return coords

    def run(self):
        """Main loop for the game."""
        while self.running:
            # Fill the background
            self.win.fill(self.BLACK)

            # Get response from the agent
            response = self.client.run(agent=self.agent, messages=self.messages)
            self.messages = response.messages
            last_message = self.messages[-1]

            # sometimes there is no content
            if last_message["content"] is not None:
                print(last_message["content"])
                try:
                    # Parse coordinates to get x1, y1, x2, y2, x3, y3
                    coords = self.parse_coordinates(last_message["content"])

                    # Draw each point or skip if out of range
                    for x, y in coords:
                        # Check if coordinates are out of range
                        if 0 <= x < 600 and 0 <= y < 400:
                            self.draw_point(x, y)
                            print(f"Point drawn at ({x}, {y})")
                        else:
                            print(f"Wrong coordinate values: ({x}, {y}) out of range")

                except ValueError:
                    print("Received invalid coordinates. Ignoring this response.")


                # Update the display
                pygame.display.update()

                # Wait for 1 second before generating a new position
                time.sleep(1)

                # Reset messages for the next iteration
                self.messages = []
            
            # if last_message["content"] is not None:
            #     # Expecting a space-separated string with x1 y1 x2 y2 x3 y3
            #     print(last_message["content"])
                # coords = list(map(int, last_message['content'].split()))

                # # Draw each point (3 points in this case)
                # for i in range(0, len(coords), 2):
                #     x, y = coords[i], coords[i + 1]
                #     self.draw_point(x, y)

                # # Update the display
                # pygame.display.update()

                # # Wait for 1 second before generating a new position
                # time.sleep(1)

                # # Reset messages for the next iteration
                # self.messages = []
            else:
                "invalid or no coordinates"
                continue

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
