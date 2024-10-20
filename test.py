from swarm import Swarm, Agent

# Initialize the Swarm client
client = Swarm()

# Create an agent with instructions
my_agent = Agent(
    name="Agent",
    instructions="respond with a number from 0 to 100",
)

def pretty_print_messages(messages):
    for message in messages:
        if message["content"] is None:
            continue
        print(f"{message['role']}: {message['content']}")

messages = []
agent = my_agent

while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})
    
    # Run the swarm with the current agent and messages
    response = client.run(agent=agent, messages=messages)
    messages = response.messages
    agent = response.agent
    
    for message in messages:
        if 'content' in message:  # Check if 'content' key exists in the message
            print("i print", message['content'])
            generated_number = message['content']

    # for message in messages:
    #     if message["role"] == "agent":  # Find the agent's response
    #         try:
    #             generated_number = int(message["content"])  # Convert the content to an integer
    #             print(f"Generated Number: {generated_number}")
    #         except ValueError:
    #             print("The agent did not return a valid number.")
    
    pretty_print_messages(messages)

    print("The number is:", generated_number)
