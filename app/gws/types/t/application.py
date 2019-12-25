### Application

from .base import Optional, List, Regex
from ..data import Config
from .db import StorageObject
from .object import Object
from .template import TemplateObject
from .misc import DocumentRootConfig


class ApiObject(Object):
    actions: dict


class ClientObject(Object):
    pass


class CorsConfig(Config):
    enabled: bool = False
    allowOrigin: str = '*'
    allowCredentials: bool = False
    allowHeaders: Optional[List[str]]


class RewriteRule(Config):
    match: Regex  #: expression to match the url against
    target: str  #: target url with placeholders
    options: Optional[dict]  #: additional options


class WebSiteObject(Object):
    host: str
    ssl: bool
    error_page: 'TemplateObject'
    static_root: 'DocumentRootConfig'
    assets_root: 'DocumentRootConfig'
    rewrite_rules: List[RewriteRule]
    reversed_rewrite_rules: List[RewriteRule]
    cors: CorsConfig

    def url_for(self, req, url: str) -> str:
        pass


class ApplicationObject(Object):
    api: ApiObject
    client: ClientObject
    qgis_version: str
    storage: 'StorageObject'
    version: str
    web_sites: List[WebSiteObject]

    def find_action(self, action_type: str, project_uid=None) -> Object:
        pass
