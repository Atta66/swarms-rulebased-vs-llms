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
        self.history = config["history"]
        self.client = Swarm()
        self.messages = []

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

    def run_agent(self, agent_config, **kwargs):
        """Run an agent to calculate its output."""
        instructions = agent_config["instructions"].format(**kwargs)
        agent = Agent(name=agent_config["name"], instructions=instructions)
        response = self.client.run(agent=agent, messages=self.messages)
        last_message = response.messages[-1]

        if last_message["content"] is not None:
            return eval(last_message["content"])
        else:
            print(f"Invalid response from {agent_config['name']}")
            return None

    def choose_path(self):
        """Use LLM to choose a path based on probabilities."""
        agent_config = {
            "name": "PathSelectionAgent",
            "instructions": "You are an ant in an Ant Colony Optimization simulation. Your task is to choose a path between two points based on the following data:\n- Paths: {paths}\n- Each path has a distance and a pheromone level.\n\nUse the formula: Probability ∝ (pheromone / distance). Calculate the probabilities for each path and return the name of the path you choose. Do not include any additional text or explanations."
        }
        return self.run_agent(agent_config, paths=self.paths)

    def update_pheromones(self, chosen_path):
        """Use LLM to update pheromone levels."""
        agent_config = {
            "name": "PheromoneUpdateAgent",
            "instructions": "You are a pheromone update agent in an Ant Colony Optimization simulation. Your task is to update the pheromone levels of the paths based on the chosen path and its quality:\n- Chosen path: {chosen_path}\n- Current pheromone levels: {pheromone_levels}\n- Quality of the chosen path: {quality} (lower distance means higher quality).\n\nIncrease the pheromone level of the chosen path based on its quality. Return the updated pheromone levels for all paths in this format: {path_name: pheromone_level}."
        }
        updated_pheromones = self.run_agent(agent_config, chosen_path=chosen_path, pheromone_levels=self.paths, quality=1 / self.paths[chosen_path]["distance"])
        if updated_pheromones:
            self.paths.update(updated_pheromones)

    def evaporate_pheromones(self):
        """Use LLM to apply pheromone evaporation."""
        agent_config = {
            "name": "EvaporationAgent",
            "instructions": "You are a pheromone evaporation agent in an Ant Colony Optimization simulation. Your task is to apply evaporation to the pheromone levels of the paths:\n- Current pheromone levels: {pheromone_levels}\n- Evaporation rate: {evaporation_rate}.\n\nDecrease the pheromone levels of all paths by the evaporation rate. Return the updated pheromone levels for all paths in this format: {path_name: pheromone_level}."
        }
        updated_pheromones = self.run_agent(agent_config, pheromone_levels=self.paths, evaporation_rate=self.evaporation_rate)
        if updated_pheromones:
            self.paths.update(updated_pheromones)

    def simulate(self, steps=50):
        for step in range(steps):
            chosen = self.choose_path()
            if chosen:
                self.history[chosen] += 1
                self.update_pheromones(chosen)
                self.evaporate_pheromones()

            # Plot the pheromone levels
            plt.clf()
            plt.bar(self.paths.keys(), [p["pheromone"] for p in self.paths.values()], color=['green', 'blue'])
            plt.title(f"Step {step+1}")
            plt.ylabel("Pheromone Level")
            plt.pause(0.2)

        plt.show()

# Initialize and run the simulation
aco = AntColonyOptimization(config_path)
aco.simulate()
