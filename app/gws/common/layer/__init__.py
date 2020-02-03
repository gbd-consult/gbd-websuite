from .layer import Layer
from .image import Image
from .imagetile import ImageTile
from .vector import Vector
from .util import add_layers_to_object

from . import layer, image, imagetile, vector, types


class Config(layer.Config):
    pass


class VectorConfig(vector.Config):
    pass


class ImageConfig(image.Config):
    pass


class ImageTileConfig(imagetile.Config):
    pass

