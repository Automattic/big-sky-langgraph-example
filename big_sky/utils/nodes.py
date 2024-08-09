from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PipelinePromptTemplate,
    MessagesPlaceholder,
)
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
import json

system_prompt = SystemMessagePromptTemplate.from_template(
    """\
{identity}
{goal}
{site}
{page}"""
)

identity_prompt = PromptTemplate.from_template("You are a chatbot.")
goal_prompt = PromptTemplate.from_template("Your goal is to help the user build a web site.")
site_prompt = PromptTemplate.from_template(
    """\
 * Title: {site.title}
 * Type: {site.type}
 * Topic: {site.topic}
 * Description: {site.description}
 * Location: {site.location}"""
)
page_prompt = PromptTemplate.from_template(
    """\
 * Title: {page.title}
 * Description: {page.description}
 * Content: {page.content}"""
)

# now compose these into a system prompt
input_prompts = [
    ("identity", identity_prompt),
    ("goal", goal_prompt),
    ("site", site_prompt),
    ("page", page_prompt),
]

final_system_prompt = PipelinePromptTemplate(
    final_prompt=system_prompt, pipeline_prompts=input_prompts
)


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
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    # site_prompt = site_prompt.format(site=state["site"])
    # system_prompt = system_prompt.format(site=site_prompt)
    # chat_prompt_template = ChatPromptTemplate.from_messages
    chat_prompt = ChatPromptTemplate.from_messages(
        [final_system_prompt, MessagesPlaceholder(variable_name="messages")]
    )
    messages = chat_prompt.format_messages(messages=messages, site=state["site"])
    # messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get("configurable", {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    response = model.invoke(messages)

    #

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
tool_node = ToolNode(tools)
