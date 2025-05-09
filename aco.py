import numpy as np
import matplotlib.pyplot as plt
import os
import json
from swarm import Agent, Swarm


base_path = "C:\\Users\\attah\\Desktop\\workspace\\ACES\\ACES"
config_file = "config.json"
config_path = os.path.join(base_path, "config", config_file)

class AntColonyOptimization:
    def __init__(self, config_path):
        self.config_path = config_path
        config = self.load_config()
        self.evaporation_rate = config["evaporation_rate"]
        self.paths = config["paths"]
        # self.history = config["history"]
        self.client = Swarm()
        self.messages = []
        self.chosen_path = None
        self.agent_prompts = [
            config["agent1"],
            config["agent2"],
            config["agent3"]
        ]
        self.pheromones = []

    def load_config(self):
        """Load configuration from the JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found.")
            raise
        except json.JSONDecodeError:
            print(f"Error decoding the config file {self.config_path}.")
            raise

    # def extract_pheromones(self, response_str):
    #     data = json.loads(response_str)
    #     return {path: info["pheromone"] for path, info in data.items()}

    def run_agent(self, agent_prompt, paths, evaporation_rate, chosen_path=None, **kwargs):
        """Run an agent to calculate its output."""
        instructions = agent_prompt["instructions"]
        print(f"Running agent: {agent_prompt['name']}")
        print(f"Instructions: {instructions}")
        
        instructions = instructions.format(
            paths=paths,
            evaporation_rate=evaporation_rate,
            chosen_path=chosen_path
        )

        agent = Agent(
            name=agent_prompt["name"],
            instructions=instructions
        )

        response = self.client.run(agent=agent, messages=self.messages)
        last_message = response.messages[-1]

        if last_message["content"] is not None:
            try:
                # Parse the response as JSON if it's expected to be a dictionary
                result = json.loads(last_message["content"].strip())
            except json.JSONDecodeError:
                print(f"Error decoding response from {agent_prompt['name']}: {last_message['content']}")
                result = last_message["content"].strip()  # Use the raw string if JSON parsing fails
        else:
            print(f"Invalid response from {agent_prompt['name']}")
            result = None  # Return None if the response is invalid

        return result

    def choose_update_paths(self):
        
        self.chosen_path = self.run_agent(self.agent_prompts[0], paths=self.paths, evaporation_rate=self.evaporation_rate)
        print(f"Chosen path: {self.chosen_path}")

        self.pheromones = self.run_agent(self.agent_prompts[1], paths=self.paths, evaporation_rate=self.evaporation_rate, chosen_path=self.chosen_path)
        print(f"Updated pheromones: {self.paths}")

        self.paths["short"]["pheromone"] = self.pheromones[0]
        self.paths["long"]["pheromone"] = self.pheromones[1]

        self.pheromones = self.run_agent(self.agent_prompts[2], paths=self.paths, evaporation_rate=self.evaporation_rate)  
        print(f"Final pheromones: {self.paths}")

        self.paths["short"]["pheromone"] = self.pheromones[0]
        self.paths["long"]["pheromone"] = self.pheromones[1]
        

    def simulate(self, steps=1):
        for step in range(steps):
            self.choose_update_paths()

            # Plot the pheromone levels
            plt.clf()
            plt.bar(self.paths.keys(), [p["pheromone"] for p in self.paths.values()], color=['green', 'blue'])
            plt.title(f"Step {step+1}")
            plt.ylabel("Pheromone Level")
            plt.pause(0.2)

        plt.show()

if __name__ == "__main__":
    aco = AntColonyOptimization(config_path)
    aco.simulate(steps=13)
    print("Simulation complete.")
    print("Final pheromone levels:", aco.paths)
    print("Path history:", aco.history)