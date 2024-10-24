from swarm import Swarm, Agent


client = Swarm()
increment = (1,2), (3,4), (5,6)

agent = Agent(
        name="Agent",
        instructions=f"""increment the x in {increment} and output in the same format as the input""",
)


# # messages = [{"role": "user", "content": "Hi!"}]
messages = []

response = client.run(agent=agent, messages=messages)

print(response.messages[-1]["content"])