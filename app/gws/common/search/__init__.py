import gws.types as t


class Config:
    enabled: bool = True  #: search is enabled
    providers: t.Optional[t.List[t.ext.search.provider.Config]]  #: list of search prodivers
