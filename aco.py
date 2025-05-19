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
        # print(f"Running agent: {agent_prompt['name']}")
        # print(f"Instructions: {instructions}")
        
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
                # print(f"Error decoding response from {agent_prompt['name']}: {last_message['content']}")
                result = last_message["content"].strip()  # Use the raw string if JSON parsing fails
        else:
            print(f"Invalid response from {agent_prompt['name']}")
            result = None  # Return None if the response is invalid

        return result

    def choose_update_paths(self):
        
        self.chosen_path = self.run_agent(self.agent_prompts[0], paths=self.paths, evaporation_rate=self.evaporation_rate)
        # print(f"Chosen path: {self.chosen_path}")

        self.pheromones = self.run_agent(self.agent_prompts[1], paths=self.paths, evaporation_rate=self.evaporation_rate, chosen_path=self.chosen_path)
        # print(f"Updated pheromones: {self.paths}")

        self.paths["short"]["pheromone"] = self.pheromones[0]
        self.paths["long"]["pheromone"] = self.pheromones[1]

        self.pheromones = self.run_agent(self.agent_prompts[2], paths=self.paths, evaporation_rate=self.evaporation_rate)  
        # print(f"Final pheromones: {self.paths}")

        self.paths["short"]["pheromone"] = self.pheromones[0]
        self.paths["long"]["pheromone"] = self.pheromones[1]
        

    def simulate(self, steps=10):
        import time
        import psutil
        import GPUtil
        cpu_usages = []
        ram_usages = []
        gpu_loads = []
        gpu_mem_usages = []
        total_time = 0.0
        for step in range(steps):
            start_time = time.time()
            self.choose_update_paths()
            step_time = time.time() - start_time
            total_time += step_time
            # System resource usage
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().percent
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_load = gpu.load * 100
                gpu_mem_usage = gpu.memoryUtil * 100
            else:
                gpu_load = 0
                gpu_mem_usage = 0
            cpu_usages.append(cpu_usage)
            ram_usages.append(ram_usage)
            gpu_loads.append(gpu_load)
            gpu_mem_usages.append(gpu_mem_usage)
            # Plot the pheromone levels
            plt.clf()
            plt.bar(self.paths.keys(), [p["pheromone"] for p in self.paths.values()], color=['green', 'blue'])
            plt.title(f"Step {step+1}")
            plt.ylabel("Pheromone Level")
            plt.pause(0.2)
        # Plot resource usage
        plt.figure(figsize=(8,4))
        plt.plot(range(steps), cpu_usages, label='CPU Usage (%)', marker='o')
        plt.plot(range(steps), ram_usages, label='RAM Usage (%)', marker='s')
        plt.plot(range(steps), gpu_loads, label='GPU Load (%)', marker='^')
        plt.plot(range(steps), gpu_mem_usages, label='GPU Mem Usage (%)', marker='x')
        plt.xlabel('Time Step')
        plt.ylabel('Usage (%)')
        plt.title('CPU, RAM, and GPU Usage Over Time (50 Steps)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(f"Total time for {steps} time steps: {total_time:.6f} seconds")
        print(f"Average CPU Usage: {sum(cpu_usages)/len(cpu_usages):.2f}%")
        print(f"Average RAM Usage: {sum(ram_usages)/len(ram_usages):.2f}%")
        print(f"Average GPU Load: {sum(gpu_loads)/len(gpu_loads):.2f}%")
        print(f"Average GPU Memory Usage: {sum(gpu_mem_usages)/len(gpu_mem_usages):.2f}%")
        print("Final pheromone levels:", self.paths)

if __name__ == "__main__":
    aco = AntColonyOptimization(config_path)
    aco.simulate(steps=50)
    print("Simulation complete.")
    print("Final pheromone levels:", aco.paths)
    print("Path history:", aco.history)