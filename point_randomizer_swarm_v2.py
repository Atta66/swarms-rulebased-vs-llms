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
    
    def parse_x_coordinate(self, content):
        # Custom method to parse the x value from the response content (expected format: "(x, y)")
        try:
            # Extract x from "(x, y)" format
            updated_x = int(content.split(",")[0].strip("()"))  # Assumes x is in the format "(x"
            return updated_x
        except ValueError:
            # Handle cases where parsing might fail (e.g., incorrect format)
            return None

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

            print("coord_string give: ", coord_string)
            
            agent2 = Agent(
                name="Agent2",
                instructions=f"""Increase the x value in {coord_string} by 20 and output the result in the exact same format as the input. Do not add any additional text.
                for example, if i have (20, 30), it will become (40, 30).""",
            )
            
            response = self.client.run(agent=agent2, messages=self.messages)
            last_message = response.messages[-1]

            if last_message["content"] is not None:
                print("Agent2 response:", last_message["content"])
                updated_coordinate = self.parse_coordinates(last_message["content"])
                updated_coordinates.extend(updated_coordinate)  # Add each updated pair to the final list

        return updated_coordinates

    # def run_agent3(self, coordinates):

    #     updated_coordinates = []

    #     for x, y in coordinates:
    #         coord_string = f"({x},{y})"

    #         # updated_coordinate = coord_string
    #         # updated_coordinates.extend(updated_coordinate)  # Add each updated pair to the final list
            
    #         agent3 = Agent(
    #         name="Agent3",
    #         instructions=f"""Check if the x-coordinate in the pair (x, y) is greater than 300. If x is less than or equal to 300, keep (x, y) the same.  
    #         Otherwise, change x to 0. Just output the result in the format (x, y) without any extra text. My pair is {coord_string}.""",
    #         )
            
    #         response = self.client.run(agent=agent3, messages=self.messages)
    #         last_message = response.messages[-1]

    #         if last_message["content"] is not None:
    #             print("Agent3 response:", last_message["content"])
    #             updated_coordinate = self.parse_coordinates(last_message["content"])
    #             updated_coordinates.extend(updated_coordinate)  # Add each updated pair to the final list
        
    #     # updated_coordinates = self.run_agent2(updated_coordinates)
    #     print("the updated coordinates are: ", updated_coordinates)
    #     return updated_coordinates
    

    def run_agent3(self, coordinates):
        updated_coordinates = []

        for x, y in coordinates:
            coord_string = f"{x}"  # Only include x in the coord_string
            print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIII: ", coord_string)

            agent3 = Agent(
                name="Agent3",
                instructions=f"""if {coord_string} is greater than 300, output 0, if it is not true, output {coord_string}""",
            )
            
            # Run the agent and get the response
            response = self.client.run(agent=agent3, messages=self.messages)
            last_message = response.messages[-1]

            # Parse the agent's response
            if last_message["content"] is not None:
                print("Agent3 response:", last_message["content"])
                
                # Parse the x value from the agent's response
                updated_x = self.parse_x_coordinate(last_message["content"])  # Adjust parse method as needed
                
                # Keep the original y value and update only x
                updated_coordinate = (updated_x, y)
                updated_coordinates.append(updated_coordinate)  # Add each updated pair to the final list

        print("The updated coordinates are:", updated_coordinates)
        return updated_coordinates

    


        # agent3 = Agent(
        #     name="Agent3",
        #     instructions=f"""Check if the x-coordinate in each of the (x, y) pairs is greater than 300. If x is less than or equal to 300, keep (x, y) the same. 
        #     Otherwise, change x to 0. Output the result in the format (x, y) without any extra text. My pairs are {coordinates}""",
        # )
        # response = self.client.run(agent=agent3, messages=self.messages)
        # last_message = response.messages[-1]
        # coordinates = self.parse_coordinates(last_message["content"])

        # coordinates = self.run_agent2(coordinates)

        # return coordinates

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
                coordinates = self.run_agent3(coordinates)

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
