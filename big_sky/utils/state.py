from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import Literal, Optional, TypedDict, Annotated, Sequence

SiteType = Literal[
    "blog",
    "store",
    "community",
    "wiki",
    "forum",
    "internal",
    "education",
    "news",
    "portfolio",
    "personal",
    "other",
]

SiteTopic = Literal[
    "art",
    "business",
    "computers",
    "games",
    "health",
    "home",
    "news",
    "recreation",
    "reference",
    "science",
    "shopping",
    "society",
    "sports",
]

class Site(TypedDict):
    type: SiteType
    title: str
    description: str
    location: str
    topic: SiteTopic

class Page(TypedDict):
    title: str
    description: str
    content: str

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    site: Optional[Site] = None
    page: Optional[Page] = None
