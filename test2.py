from swarm import Swarm, Agent

class SwarmManager:
    def __init__(self):
        # Initialize the Swarm client
        self.client = Swarm()

        # Create agents with specific instructions
        self.agent1 = Agent(
            name="Agent 1",
            instructions="respond with a number from 0 to 20",
        )
        
        self.agent2 = Agent(
            name="Agent 2",
            instructions="respond with a number from 10 to 20",
        )
        
        self.agent3 = Agent(
            name="Agent 3",
            instructions="check if the numbers from Agent 1 and Agent 2 match",
        )

        self.messages = []

    def pretty_print_messages(self, messages):
        for message in messages:
            if message["content"] is None:
                continue
            print(f"{message['role']}: {message['content']}")

    def run_agents(self):
        # Initialize messages for interaction
        self.messages.append({"role": "user", "content": "num"})

        # Run Agent 1
        response = self.client.run(agent=self.agent1, messages=self.messages)
        self.messages = response.messages
        self.agent1 = response.agent

        for message in self.messages:
            if 'content' in message:
                generated_number1 = int(message['content'])  # Agent 1's number
                print("Agent 1 generated:", generated_number1)

        # Run Agent 2
        self.messages.append({"role": "user", "content": "num"})
        response = self.client.run(agent=self.agent2, messages=self.messages)
        self.messages = response.messages
        self.agent2 = response.agent

        for message in self.messages:
            if 'content' in message:
                generated_number2 = int(message['content'])  # Agent 2's number
                print("Agent 2 generated:", generated_number2)

        # Run Agent 3 to check if numbers match
        self.messages.append({"role": "user", "content": f"Check if {generated_number1} matches {generated_number2}"})
        response = self.client.run(agent=self.agent3, messages=self.messages)
        self.messages = response.messages
        self.agent3 = response.agent

        for message in self.messages:
            if 'content' in message:
                print("Agent 3 response:", message['content'])

        # Print all messages for clarity
        self.pretty_print_messages(self.messages)


# Create an instance of SwarmManager and run the agents
if __name__ == "__main__":
    swarm_manager = SwarmManager()
    swarm_manager.run_agents()
