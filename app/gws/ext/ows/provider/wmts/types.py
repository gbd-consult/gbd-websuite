import gws.types as t


class TileMatrix(t.Data):
    uid: str
    scale: float
    x: float
    y: float
    width: float
    height: float
    tile_width: float
    tile_height: float
    extent: t.Extent


class TileMatrixSet(t.Data):
    uid: str
    crs: t.Crs
    matrices: t.List[TileMatrix]


class SourceLayer(t.SourceLayer):
    matrix_sets: t.List[TileMatrixSet]
    matrix_ids: t.List[str]
    format: str
