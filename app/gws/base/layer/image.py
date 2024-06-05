"""Base image layer."""

from . import base


class Object(base.Object):
    """Base image layer"""

    canRenderBox = True
    canRenderXyz = True
    supportsVectorOws = True
