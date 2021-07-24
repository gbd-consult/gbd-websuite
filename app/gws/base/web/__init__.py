import gws
import gws.types as t

from .core import DocumentRootConfig, SSLConfig, document_root_from_config
from . import site


class Config(gws.Config):
    """Web server configuration"""

    sites: t.Optional[t.List[site.Config]]  #: configured sites
    ssl: t.Optional[SSLConfig]  #: ssl configuration


DEFAULT_SITE = Config(
    host='*',
    root=DocumentRootConfig(dir='/data/web')
)
