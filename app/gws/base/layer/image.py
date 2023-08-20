"""Base image layer."""

from . import core


class Object(core.Object):
    """Base image layer"""

    canRenderBox = True
    canRenderXyz = True
    supportsVectorOws = True
