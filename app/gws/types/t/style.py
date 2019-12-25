### Styles

from .base import Optional
from ..data import Config, Props


class StyleProps(Props):
    type: str
    content: Optional[dict]
    text: Optional[str]


class StyleConfig(Config):
    """Feature style"""

    type: str  #: style type ("css")
    content: Optional[dict]  #: css rules
    text: Optional[str]  #: raw style content
