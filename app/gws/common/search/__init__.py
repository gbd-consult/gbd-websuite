import gws.types as t


class Config:
    enabled: bool = True  #: search is enabled
    providers: t.Optional[t.List[t.ext.search.provider.Config]]  #: search prodivers


#:export
class SearchArgs(t.Data):
    axis: str
    bounds: t.Bounds
    keyword: t.Optional[str]
    filter: t.Optional[t.SearchFilter]
    layers: t.List[t.ILayer]
    limit: int
    params: dict
    project: t.IProject
    resolution: float
    shapes: t.List[t.IShape]
    source_layer_names: t.List[str]
    tolerance: t.Measurement
    relation_depth: int
