import gws.types as t


class TileMatrix:
    def __init__(self):
        self.uid: str = ''
        self.scale: float = 0
        self.x: float = 0
        self.y: float = 0
        self.width: float = 0
        self.height: float = 0
        self.tile_width: float = 0
        self.tile_height: float = 0
        self.extent = []


class TileMatrixSet:
    def __init__(self):
        self.uid: str = ''
        self.crs: str = ''
        self.matrices: t.List[TileMatrix] = []


class SourceLayer(t.SourceLayer):
    def __init__(self):
        super().__init__()
        self.matrix_sets: t.List[TileMatrixSet] = []
        self.matrix_ids: t.List[str] = []
        self.format: str = []
