import pygame
import time
from swarm import Swarm, Agent
import re
import math

class RandomMovingPoints:
    def __init__(self, width=600, height=400, point_size=5, num_points=5, radius=400):
        # Initialize pygame
        pygame.init()
        self.client = Swarm()
        self.width = width
        self.height = height
        self.point_size = point_size
        self.num_points = num_points  # Number of points to draw
        self.radius = radius  # Radius within which points will check for each other
        self.win = pygame.display.set_mode((self.width, self.height))
        self.initial_step = 0
        pygame.display.set_caption("Random Moving Points")

        # Define colors
        self.BLACK = (0, 0, 0)
        self.POINT_COLOR = (255, 0, 0)  # Red points

        # Set up agent with instructions for 3 points
        self.agent1 = Agent(
            name="Agent1",
            instructions=f"""Generate exactly {self.num_points} pairs of (x, y) coordinates in this format: 
            (x1,y1), (x2,y2), (x3,y3). For each pair, x should be between 0 and {self.width - 1}, and y 
            should be between 0 and {self.height - 1}. Provide only the coordinates, with no additional text.""",
        )

        self.messages = []

        # Initialize pygame clock for frame rate control
        self.clock = pygame.time.Clock()
        self.running = True

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
        response = self.client.run(agent=self.agent1, messages=self.messages)
        self.messages = response.messages
        last_message = self.messages[-1]

        if last_message["content"] is not None:
            coordinates = self.parse_coordinates(last_message["content"])
        else:
            print("Invalid response from Agent1")
        print("generated coordinates (agent1): ", coordinates)
        return coordinates

    def run_agent2(self, coordinates):
        """Run Agent2 in the loop to increment the coordinates."""
        agent2 = Agent(
            name="Agent2",
            # both in positive and negative direction
            # instructions=f"""For each coordinate in {coordinates}, increase or 
            # decrease the x value by a random amount between -30 and -15, or between 15 and 
            # 30, and the y value by a random amount between 30 and 50. Provide the updated 
            # coordinates in the same format as the input, with no additional text.""",
            
            # only in positive direction
            # instructions=f"""For each coordinate in {coordinates}, increase x value by 
            # a random amount between 15 and 30, and the y value by a random amount between 
            # 30 and 50. Provide the updated coordinates in the same format as the input, 
            # with no additional text.""",

            # specification that each point should have a different random value
            instructions=f"""For each coordinate in {coordinates}, increase the x-coordinate 
            by a random amount between 15 and 30, and the y-coordinate by a random amount between 
            30 and 50. Ensure that each point gets a different random value for its x and y. Return 
            the updated coordinates in the same format as the input, with no additional text.""",
        )
        response = self.client.run(agent=agent2, messages=self.messages)
        last_message = response.messages[-1]

        if last_message["content"] is not None:
            coordinates = self.parse_coordinates(last_message["content"])
        print("incremented coordinates (agent2): ", coordinates)
        return coordinates

    def run_agent3(self, coordinates):
        """Update y values greater than 300 to 0."""
        agent3 = Agent(
            name="Agent3",
            instructions=f"""Update only the y values in {coordinates} that are greater than 300 by 
            setting them to 0 and update only the x values in {coordinates} that are greater than 500
            by setting them to 0. Return the result in the same format as {coordinates}, with no extra 
            text and keeping the same number of coordinate pairs as in the input.""",
        )
        response = self.client.run(agent=agent3, messages=self.messages)
        last_message = response.messages[-1]
        coordinates = self.parse_coordinates(last_message["content"])
        print("boundary check coordinates (agent3): ", coordinates)
        return coordinates

    # def run_agent4(self, coordinates):
    #     """Check each point's radius and move in the -x direction if another point is inside its radius."""
    #     updated_coordinates = []
    #     for i, (x1, y1) in enumerate(coordinates):
    #         for j, (x2, y2) in enumerate(coordinates):
    #             if i != j:  # Don't compare the point with itself
    #                 # Calculate the Euclidean distance between points
    #                 distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    #                 if distance < self.radius:
    #                     # Move point in the -x direction if within radius
    #                     x1 -= 50  # Move by 50 units in the -x direction (or adjust as needed)
    #         updated_coordinates.append((x1, y1))
    #     return updated_coordinates

    def run_agent4(self, coordinates):
        """Use Agent4 to check each point's radius and move in the -x direction if another point is inside its radius."""
        # Create the agent with the appropriate instructions for checking distances
        agent4 = Agent(
            name="Agent4",
            instructions=f"""For each coordinate in {coordinates}, check if any other coordinate is within a radius of {self.radius}.
            If the distance between two points is smaller than the radius, adjust the x-coordinate of the first point by moving it -50 units. 
            Return the updated coordinates in the same format as the input, with no additional text.""",
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

      
        # """What agent4 does"""

        # Define the list of points
        # points = [(47, 331), (247, 380), (162, 105), (137, 370), (44, 270)]
        # radius = 30
        # adjustment = -50

        # # Function to check if two points are within the given radius
        # def is_within_radius(p1, p2, radius):
        #     return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) <= radius

        # # Check each point for neighbors within the radius and adjust x-coordinate if necessary
        # adjusted_points = []
        # for i, p1 in enumerate(points):
        #     adjust = False
        #     for j, p2 in enumerate(points):
        #         if i != j and is_within_radius(p1, p2, radius):
        #             adjust = True
        #             break
        #     if adjust:
        #         adjusted_points.append((p1[0] + adjustment, p1[1]))
        #     else:
        #         adjusted_points.append(p1)

        # adjusted_points

    def draw_and_update_points(self, coordinates):
        """Draw each point on the screen if it is within range, update display, and pause."""
        # Clear the background
        self.win.fill(self.BLACK)
        
        # Draw each point or skip if out of range
        for x, y in coordinates:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.draw_point(x, y)

        # Update the display
        pygame.display.update()

        # Wait for 1 second before generating a new position
        time.sleep(0.5)



    def run(self):
        coordinates = []
        
        while self.running:
            # Fill the background
            self.win.fill(self.BLACK)

            if self.initial_step == 0:
                # Get response from agent 1
                coordinates = self.run_agent1()
                self.initial_step += 1
                self.draw_and_update_points(coordinates)
            else:
                # Get response from agent 3 (including agent 2 steps)
                coordinates = self.run_agent3(coordinates)
                self.draw_and_update_points(coordinates)

                coordinates = self.run_agent2(coordinates)
                self.draw_and_update_points(coordinates)

                # Run Agent4 to check if points are within radius of each other
                coordinates = self.run_agent4(coordinates)
                self.draw_and_update_points(coordinates)

            
            # # Draw each point or skip if out of range
            # for x, y in coordinates:
            #     # Check if coordinates are out of range
            #     if 0 <= x < self.width and 0 <= y < self.height:
            #         self.draw_point(x, y)

            # # Update the display
            # pygame.display.update()

            # # Wait for 1 second before generating a new position
            # time.sleep(1)

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