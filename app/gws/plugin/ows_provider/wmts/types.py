import gws
import gws.types as t


class TileMatrix(gws.Data):
    uid: str
    scale: float
    x: float
    y: float
    width: float
    height: float
    tile_width: float
    tile_height: float
    extent: gws.Extent


class TileMatrixSet(gws.Data):
    uid: str
    crs: gws.Crs
    matrices: t.List[TileMatrix]


class SourceLayer(gws.SourceLayer):
    matrix_sets: t.List[TileMatrixSet]
    matrix_ids: t.List[str]
    format: str
