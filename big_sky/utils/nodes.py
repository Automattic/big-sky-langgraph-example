from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    BasePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import ToolMessage
from .tools import tools
from langgraph.prebuilt import ToolNode


class ExtendedSystemMessagePromptTemplate(SystemMessagePromptTemplate):
    prompt: BasePromptTemplate


# human_message_prompt = ExtendedHumanMessagePromptTemplate(prompt=pipeline_prompt)

tmp_system_prompt = PromptTemplate.from_template(
    """\
    You are Wapuu, a chatbot that helps users build web sites.
    Your goal is to help the user build a web site.

    Site Information:
    * Title: {{site.title}}
    * Type: {{site.type}}
    * Topic: {{site.topic}}
    * Description: {{site.description}}
    * Location: {{site.location}}

    Page Information:
    * Title: {{page.title}}
    * Description: {{page.description}}
    * Content: {{page.content}}""",
    template_format="mustache",
)

# system_prompt = PromptTemplate.from_template(
#     """\
# {{identity}}
# {{goal}}
# {{site}}
# {{page}}""",
#     template_format="mustache",
# )

# identity_prompt = PromptTemplate.from_template(
#     "You are Wapuu, a chatbot that helps users build web sites.",
#     template_format="mustache",
# )

# goal_prompt = PromptTemplate.from_template(
#     "Your goal is to help the user build a web site.", template_format="mustache"
# )

# site_prompt = PromptTemplate.from_template(
#     """\
#  * Title: {{site.title}}
#  * Type: {{site.type}}
#  * Topic: {{site.topic}}
#  * Description: {{site.description}}
#  * Location: {{site.location}}""",
#     template_format="mustache",
# )

# page_prompt = PromptTemplate.from_template(
#     """\
#  * Title: {{page.title}}
#  * Description: {{page.description}}
#  * Content: {{page.content}}""",
#     template_format="mustache",
# )

# # now compose these into a system prompt
# input_prompts = [
#     ("identity", identity_prompt),
#     ("goal", goal_prompt),
#     ("site", site_prompt),
#     ("page", page_prompt),
# ]

# final_system_prompt = PipelinePromptTemplate(
#     final_prompt=system_prompt, pipeline_prompts=input_prompts
# )


# print(final_system_prompt)


class UpdateStateFromToolCall:
    """Updates the site if there are relevant tool calls present in the tail of the messages list."""

    def __init__(self) -> None:
        pass

    def __call__(self, inputs: dict):
        # Ensure 'site' is initialized as a dictionary if it's None
        outputs = {"site": inputs.get("site", {})}

        if messages := inputs.get("messages", []):
            # Find the highest index of a non-ToolMessage
            highest_non_tool_index = -1
            for i, message in enumerate(messages):
                if not isinstance(message, ToolMessage):
                    highest_non_tool_index = i

            # Extract the slice of ToolMessages starting after the highest non-ToolMessage index
            tool_messages_tail = messages[highest_non_tool_index + 1 :]

            # Process the ToolMessages in order
            for message in tool_messages_tail:
                if isinstance(message.artifact, dict):
                    # Update the outputs with the site artifact
                    outputs.update(message.artifact)

            return outputs
        else:
            raise ValueError("No messages found in input")


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model_name == "anthropic":
        model = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state.get("messages", [])
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


# Define the function that calls the model
def call_model(state, config):
    messages = state.get("messages", [])
    # site_prompt = site_prompt.format(site=state["site"])
    # system_prompt = system_prompt.format(site=site_prompt)
    # chat_prompt_template = ChatPromptTemplate.from_messages
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ExtendedSystemMessagePromptTemplate(prompt=tmp_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    messages = chat_prompt.format_messages(messages=messages)
    # messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get("configurable", {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

update_state_node = UpdateStateFromToolCall()

# Define the function to execute tools
tool_node = ToolNode(tools)
