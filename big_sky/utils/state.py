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


class Site(TypedDict, total=False):
    type: Optional[SiteType]
    title: Optional[str]
    description: Optional[str]
    location: Optional[str]
    topic: Optional[SiteTopic]


class Page(TypedDict, total=False):
    title: Optional[str]
    description: Optional[str]
    content: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    site: Optional[Site] = None
    page: Optional[Page] = None
