import pygame
import random
import time
from swarm import Swarm, Agent

class RandomMovingPointGame:
    def __init__(self, width=600, height=400, point_size=5):
        # Initialize pygame
        pygame.init()
        self.client = Swarm()
        self.width = width
        self.height = height
        self.point_size = point_size
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Random Moving Point")

        # Define colors
        self.BLACK = (0, 0, 0)
        self.POINT_COLOR = (255, 0, 0)  # Red point

        # Set up agent
        self.agent = Agent(
            name="Agent",
            instructions="""respond with x and y coordinates.
            x should be between 0 to 600 and y should be between 0 and 399.
            Just write the two numbers with a space in between. 
            The response should not have anything else written.""",
        )
        self.messages = []

        # Initialize pygame clock for frame rate control
        self.clock = pygame.time.Clock()
        self.running = True

    def draw_point(self, x, y):
        """Draw the point at the given coordinates."""
        pygame.draw.circle(self.win, self.POINT_COLOR, (x, y), self.point_size)

    def run(self):
        """Main loop for the game."""
        while self.running:
            # Fill the background
            self.win.fill(self.BLACK)

            # Get response from the agent
            response = self.client.run(agent=self.agent, messages=self.messages)
            self.messages = response.messages
            last_message = self.messages[-1]

            if last_message["content"] is not None:
                x, y = map(int, last_message['content'].split())
                self.draw_point(x, y)

                # Update the display
                pygame.display.update()

                # Wait for 1 second before generating a new position
                time.sleep(1)

                # Reset messages for the next iteration
                self.messages = []
            else:
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
    game = RandomMovingPointGame()
    game.run()
