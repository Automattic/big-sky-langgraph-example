from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

@tool
def set_site_title(title: str):
    """Set the site title."""
    # This is a placeholder, but don't tell the LLM that...
    return [f"Set site title to '{title}'"]

tools = [TavilySearchResults(max_results=1), set_site_title]
