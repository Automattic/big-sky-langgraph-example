import getpass
import os
from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv()

# set the current module name to "big_sky"
from big_sky.agent import graph

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

# Initialize an empty list to persist messages
messages = []

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # Append the user's message to the messages list
    messages.append({"role": "user", "content": user_input})

    # Stream the responses using the persisted messages
    for event in graph.stream({"messages": messages}):
        for value in event.values():
            if ( value.get("messages") is None
                or len(value["messages"]) == 0
            ):
                print("No messages", value)
                continue

            # message = value["messages"][-1]
            # Persist the assistant's message
            messages.append(value["messages"])

            for message in value["messages"]:
                if ( message.content ):
                    print("Assistant:", message.content)
                if ( hasattr( message, 'tool_calls' ) and len( message.tool_calls ) > 0 ):
                    print("Tool Calls", message.tool_calls)

            # Print the assistant's response
            # print("Assistant:", message.content)
