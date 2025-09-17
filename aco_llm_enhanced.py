#!/usr/bin/env python3
"""
Enhanced ACO LLM with performance tracking and same interface as regular ACO
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import psutil
import GPUtil
from datetime import datetime
import random
from swarm import Agent, Swarm


class ACOLLMEnhanced:
    def __init__(self, config_file="config/config.json"):
        self.config_file = config_file
        self.load_config()
        self.client = Swarm()
        self.messages = []
        
        # Initialize paths
        self.paths = {
            "short": {
                "distance": self.config["paths"]["short"]["distance"],
                "pheromone": self.config["paths"]["short"]["pheromone"]
            },
            "long": {
                "distance": self.config["paths"]["long"]["distance"], 
                "pheromone": self.config["paths"]["long"]["pheromone"]
            }
        }
        
        # Tracking variables for performance analysis
        self.pheromone_history = []
        self.path_selection_history = {"short": [], "long": []}
        self.step_wise_ratios = []
        
    def load_config(self):
        """Load configuration from the JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found.")
            raise
        except json.JSONDecodeError:
            print(f"Error decoding the config file {self.config_file}.")
            raise

    def run_agent(self, agent_prompt, **kwargs):
        """Run an agent to calculate its output."""
        instructions = agent_prompt["instructions"]
        
        # Format instructions with provided kwargs
        instructions = instructions.format(
            paths=self.paths,
            evaporation_rate=self.config["evaporation_rate"],
            **kwargs
        )

        agent = Agent(
            name=agent_prompt["name"],
            instructions=instructions
        )

        response = self.client.run(agent=agent, messages=[])
        last_message = response.messages[-1]

        if last_message["content"] is not None:
            try:
                # Try to parse as JSON first
                if last_message["content"].strip().startswith('['):
                    result = json.loads(last_message["content"].strip())
                else:
                    result = last_message["content"].strip()
            except json.JSONDecodeError:
                result = last_message["content"].strip()
        else:
            print(f"Invalid response from {agent_prompt['name']}")
            result = None

        return result

    def choose_path(self, step):
        """Choose a path using LLM agent"""
        try:
            chosen_path = self.run_agent(self.config["agent1"])
            
            # Clean up the response to ensure it's just "short" or "long"
            if isinstance(chosen_path, str):
                chosen_path = chosen_path.lower().strip()
                if "short" in chosen_path:
                    chosen_path = "short"
                elif "long" in chosen_path:
                    chosen_path = "long"
                else:
                    # Fallback to probabilistic choice if LLM response is unclear
                    chosen_path = self.probabilistic_choice()
            else:
                chosen_path = self.probabilistic_choice()
                
            # Track the selection
            if chosen_path in ["short", "long"]:
                self.path_selection_history[chosen_path].append(step)
            
            return chosen_path
            
        except Exception as e:
            print(f"Error in path selection: {e}")
            return self.probabilistic_choice()
    
    def probabilistic_choice(self):
        """Fallback probabilistic path choice based on pheromones"""
        total = sum(p["pheromone"] / p["distance"] for p in self.paths.values())
        if total == 0:
            return "short"  # Default fallback
        
        probs = {name: (p["pheromone"] / p["distance"]) / total for name, p in self.paths.items()}
        return np.random.choice(list(probs.keys()), p=list(probs.values()))

    def update_pheromones(self, chosen_path):
        """Update pheromones using LLM agents"""
        try:
            # Agent 2: Update pheromones based on chosen path
            pheromone_update = self.run_agent(self.config["agent2"], chosen_path=chosen_path)
            
            if isinstance(pheromone_update, list) and len(pheromone_update) == 2:
                # Apply pheromone update
                if chosen_path == "short":
                    self.paths["short"]["pheromone"] += 1 / self.paths["short"]["distance"]
                else:
                    self.paths["long"]["pheromone"] += 1 / self.paths["long"]["distance"]
                
                # Use LLM suggested values if reasonable
                if all(isinstance(x, (int, float)) and x >= 0 for x in pheromone_update):
                    self.paths["short"]["pheromone"] = max(0.1, float(pheromone_update[0]))
                    self.paths["long"]["pheromone"] = max(0.1, float(pheromone_update[1]))
            else:
                # Fallback to standard ACO update
                self.paths[chosen_path]["pheromone"] += 1 / self.paths[chosen_path]["distance"]
                
        except Exception as e:
            print(f"Error in pheromone update: {e}")
            # Fallback to standard ACO update
            self.paths[chosen_path]["pheromone"] += 1 / self.paths[chosen_path]["distance"]

    def apply_evaporation(self):
        """Apply evaporation using LLM agent"""
        try:
            # Agent 3: Apply evaporation
            evaporation_result = self.run_agent(self.config["agent3"])
            
            if isinstance(evaporation_result, list) and len(evaporation_result) == 2:
                # Use LLM suggested evaporation if reasonable
                if all(isinstance(x, (int, float)) and x >= 0 for x in evaporation_result):
                    self.paths["short"]["pheromone"] = max(0.1, float(evaporation_result[0]))
                    self.paths["long"]["pheromone"] = max(0.1, float(evaporation_result[1]))
                else:
                    # Fallback to standard evaporation
                    for path in self.paths.values():
                        path["pheromone"] *= (1 - self.config["evaporation_rate"])
            else:
                # Fallback to standard evaporation
                for path in self.paths.values():
                    path["pheromone"] *= (1 - self.config["evaporation_rate"])
                    
        except Exception as e:
            print(f"Error in evaporation: {e}")
            # Fallback to standard evaporation
            for path in self.paths.values():
                path["pheromone"] *= (1 - self.config["evaporation_rate"])

    def simulate(self, max_iterations=15, target_ratio=None, ratio_tolerance=None, seed=None, show_visualization=False):
        """
        Run ACO LLM simulation with same interface as regular ACO
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Set defaults if not provided
        if target_ratio is None:
            target_ratio = 10.0
        if ratio_tolerance is None:
            ratio_tolerance = 1.0
            
        # Reset tracking variables
        self.pheromone_history = []
        self.path_selection_history = {"short": [], "long": []}
        self.step_wise_ratios = []
        
        # Run simulation
        for step in range(max_iterations):
            # Choose path using LLM
            chosen_path = self.choose_path(step)
            
            # Update pheromones using LLM
            self.update_pheromones(chosen_path)
            
            # Apply evaporation using LLM
            self.apply_evaporation()
            
            # Store pheromone state for analysis
            self.pheromone_history.append({
                "short": self.paths["short"]["pheromone"],
                "long": self.paths["long"]["pheromone"],
                "step": step
            })
            
            # Calculate and store ratio
            if self.paths["long"]["pheromone"] > 0:
                ratio = self.paths["short"]["pheromone"] / self.paths["long"]["pheromone"]
                self.step_wise_ratios.append(ratio)
            else:
                self.step_wise_ratios.append(float('inf'))
            
            # Optional visualization
            if show_visualization:
                plt.clf()
                plt.bar(self.paths.keys(), [p["pheromone"] for p in self.paths.values()], 
                       color=['green', 'blue'])
                plt.title(f"ACO LLM - Step {step+1}")
                plt.ylabel("Pheromone Level")
                plt.pause(0.2)
            
            # Check for early convergence
            if self.paths["long"]["pheromone"] > 0:
                current_ratio = self.paths["short"]["pheromone"] / self.paths["long"]["pheromone"]
                if abs(current_ratio - target_ratio) < ratio_tolerance:
                    return step + 1
        
        return max_iterations

    def get_results(self):
        """Get simulation results in the same format as regular ACO"""
        return {
            'final_pheromone_levels': self.paths.copy(),
            'pheromone_history': self.pheromone_history.copy(),
            'path_selection_history': self.path_selection_history.copy(),
            'step_wise_ratios': self.step_wise_ratios.copy(),
            'total_path_selections': {
                'short': len(self.path_selection_history["short"]),
                'long': len(self.path_selection_history["long"])
            }
        }

    def get_system_performance(self):
        """Get current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU monitoring
            gpu_load = 0
            gpu_memory = 0
            gpu_temp = 0
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_load = gpu.load * 100
                    gpu_memory = gpu.memoryUtil * 100
                    gpu_temp = gpu.temperature
            except:
                pass
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory.used / (1024 * 1024),
                'memory_percent': memory.percent,
                'gpu_load': gpu_load,
                'gpu_memory': gpu_memory,
                'gpu_temp': gpu_temp,
                'threads': psutil.cpu_count()
            }
        except Exception as e:
            print(f"⚠️  Performance monitoring error: {e}")
            return {
                'cpu_percent': 0, 'memory_mb': 0, 'memory_percent': 0,
                'gpu_load': 0, 'gpu_memory': 0, 'gpu_temp': 0, 'threads': 1
            }


def main():
    """Test the enhanced ACO LLM"""
    print("🤖 Testing Enhanced ACO LLM")
    print("=" * 50)
    
    try:
        aco_llm = ACOLLMEnhanced()
        
        # Run a test simulation
        steps_taken = aco_llm.simulate(max_iterations=15, seed=12345, show_visualization=False)
        
        # Get results
        results = aco_llm.get_results()
        
        print(f"\n✅ Simulation completed in {steps_taken} steps")
        print(f"Final pheromone levels: {results['final_pheromone_levels']}")
        print(f"Path selections - Short: {results['total_path_selections']['short']}, "
              f"Long: {results['total_path_selections']['long']}")
        
        if results['step_wise_ratios']:
            final_ratio = results['step_wise_ratios'][-1]
            print(f"Final ratio (short/long): {final_ratio:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
