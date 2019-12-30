import gws.types as t


class Config:
    enabled: bool = True  #: search is enabled
    providers: t.Optional[t.List[t.ext.search.provider.Config]]  #: list of search prodivers


#:export
class SearchArgs(t.Data):
    axis: str
    bbox: t.Extent
    count: int
    crs: t.Crs
    feature_format: t.IFormat
    keyword: t.Optional[str]
    layers: t.List[t.ILayer]
    limit: int
    params: dict
    point: t.Point
    project: t.IProject
    resolution: float
    shapes: t.List[t.IShape]
    tolerance: int
