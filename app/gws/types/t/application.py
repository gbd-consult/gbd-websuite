### Application

from .base import Optional, List, Regex
from ..data import Config


class CorsConfig(Config):
    enabled: bool = False
    allowOrigin: str = '*'
    allowCredentials: bool = False
    allowHeaders: Optional[List[str]]


class RewriteRule(Config):
    match: Regex  #: expression to match the url against
    target: str  #: target url with placeholders
    options: Optional[dict]  #: additional options
