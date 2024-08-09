from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolNode
from typing_extensions import Annotated, TypedDict
from typing import Any, List, Tuple

from .state import Site


@tool(response_format="content_and_artifact")
def set_site_title(title: str, site: Annotated[Site, InjectedState]):
    """Update the site title."""
    return f"Set site title to '{title}'", {
        "site": {
            "title": title,
        }
    }


@tool(response_format="content_and_artifact")
def set_site_topic(topic: str, site: Annotated[Site, InjectedState]):
    """Update the site topic."""
    return f"Set site topic to '{topic}'", {
        "site": {
            "topic": topic,
        }
    }


@tool(response_format="content_and_artifact")
def set_site_type(type: str, site: Annotated[Site, InjectedState]):
    """Update the site type."""
    return f"Set site type to '{type}'", {
        "site": {
            "type": type,
        }
    }


@tool(response_format="content_and_artifact")
def set_site_description(description: str, site: Annotated[Site, InjectedState]):
    """Update the site description."""
    return f"Set site description to '{description}'", {
        "site": {
            "description": description,
        }
    }


@tool(response_format="content_and_artifact")
def set_site_location(location: str, site: Annotated[Site, InjectedState]):
    """Update the site location."""
    return f"Set site location to '{location}'", {
        "site": {
            "location": location,
        }
    }


tools = [
    TavilySearchResults(max_results=1),
    set_site_title,
    set_site_topic,
    set_site_type,
    set_site_description,
    set_site_location,
]
