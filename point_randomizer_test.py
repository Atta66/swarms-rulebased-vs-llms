from swarm import Swarm, Agent
import time
# Initialize pygame
client = Swarm()
messages = []

my_agent = Agent(
    name="Agent",
    instructions="respond with x and y coordinates, x should be between 0 to 599 and y should be between 0 and 399, just write the two numbers with a space in between, the response should not have anything else written",
)

# response = client.run(my_agent, "Generate coordinates").message

while True:

    response = client.run(agent=my_agent, messages=messages)
    messages = response.messages
    last_message = messages[-1]
    if last_message["content"] is None:
            continue
    else:  
        # print(messages)
        print(last_message['content'])
        # response = "10 20"
        # x, y = map(int, last_message['content'].split())

        # print(x,y)
        messages = []
        last_message = []
        time.sleep(2)
        