from swarm import Swarm, Agent

client = Swarm()

def transfer_to_agent_b():
    global agent  # Ensure we can modify the `agent` variable
    print("Transferring to Agent B")
    agent = agent_b  # Switch to Agent B
    return {"role": "system", "content": "You are now speaking with Agent B."}  # Return a system message

agent_a = Agent(
    name="Agent A",
    instructions="You are a helpful agent.",
    functions=[transfer_to_agent_b],  # Function to switch to Agent B
)

agent_b = Agent(
    name="Agent B",
    instructions="Only speak in Haikus.",
)

# Start with Agent A
agent = agent_a
messages = []

def pretty_print_messages(messages):
    for message in messages:
        if message["content"] is None:
            continue
        print(f"{message['sender']}: {message['content']}")

while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})
    
    # Call client.run with the current agent
    response = client.run(agent=agent, messages=messages)
    
    # Update messages from the response
    messages = response.messages
    
    # Check if the agent should be switched after the response
    if agent == agent_a and "talk to agent b" in user_input.lower():
        transfer_to_agent_b()  # Switch to Agent B
    
    pretty_print_messages(messages)
