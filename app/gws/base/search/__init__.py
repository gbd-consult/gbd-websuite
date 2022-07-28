import gws
import gws.types as t

from . import provider


class Config(gws.Config):
    enabled: bool = True  #: search is enabled
    providers: t.Optional[t.List[gws.ext.config.finder]]  #: search prodivers
