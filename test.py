import json
import time
import psutil
import GPUtil
from swarm import Swarm, Agent

# Function to monitor system resources (only single reading)
def get_system_resources():
    cpu_usage = psutil.cpu_percent(interval=None)  # CPU usage as a percentage
    ram_usage = psutil.virtual_memory().percent  # RAM usage as a percentage

    # Check GPU usage (if available)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_load = gpu.load * 100  # GPU load as a percentage
        gpu_mem_usage = gpu.memoryUtil * 100  # GPU memory usage as a percentage
    else:
        gpu_load = 0
        gpu_mem_usage = 0

    return cpu_usage, ram_usage, gpu_load, gpu_mem_usage

# Function to load the configuration from the provided path
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Function to simulate the agent path selection with LLM
def select_path_with_llm(config_path):
    config = load_config(config_path)
    paths = config["paths"]
    agent_prompt = config["agent1"]
    evaporation_rate = config.get("evaporation_rate", 0.01)

    client = Swarm()
    instructions = agent_prompt["instructions"].format(
        paths=paths,
        evaporation_rate=evaporation_rate,
        chosen_path=None
    )
    print("Instructions sent to LLM:\n", instructions)  # Print the instruction

    agent = Agent(
        name=agent_prompt["name"],
        instructions=instructions
    )

    start_time = time.time()
    response = client.run(agent=agent, messages=[])

    latency = time.time() - start_time
    last_message = response.messages[-1]
    if last_message["content"] is not None:
        result = last_message["content"].strip()
        print("LLM chose path:", result)
        print(f"Latency: {latency:.3f} seconds")
    else:
        print("No valid response from LLM.")
        result = None
        print(f"Latency: {latency:.3f} seconds")

    # Get system resources after the LLM decision
    cpu_usage, ram_usage, gpu_load, gpu_mem_usage = get_system_resources()

    # Print the resource usage values
    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"RAM Usage: {ram_usage:.2f}%")
    print(f"GPU Load: {gpu_load:.2f}%")
    print(f"GPU Memory Usage: {gpu_mem_usage:.2f}%")

    return result

if __name__ == "__main__":
    config_path = "config/config.json"  # Adjust path if needed
    select_path_with_llm(config_path)
