"""Base layer object."""

from typing import Optional

import gws
import gws.gis.bounds
import gws.gis.source
import gws.lib.xmlx

DEFAULT_TILE_SIZE = 256


class CacheConfig(gws.Config):
    """Cache configuration"""

    maxAge: gws.Duration = '7d'
    """cache max. age"""
    maxLevel: int = 1
    """max. zoom level to cache"""
    requestBuffer: Optional[int]
    requestTiles: Optional[int]


class GridConfig(gws.Config):
    """Grid configuration for caches and tiled data"""

    crs: Optional[gws.CrsName]
    extent: Optional[gws.Extent]
    origin: Optional[gws.Origin]
    resolutions: Optional[list[float]]
    tileSize: Optional[int]


class AutoLayersOptions(gws.ConfigWithAccess):
    """Configuration for automatic layers."""

    applyTo: Optional[gws.gis.source.LayerFilter]
    config: dict


class ClientOptions(gws.Data):
    """Client options for a layer"""

    expanded: bool = False
    """the layer is expanded in the list view"""
    unlisted: bool = False
    """the layer is hidden in the list view"""
    selected: bool = False
    """the layer is initially selected"""
    hidden: bool = False
    """the layer is initially hidden"""
    unfolded: bool = False
    """the layer is not listed, but its children are"""
    exclusive: bool = False
    """only one of this layer's children is visible at a time"""

class GridProps(gws.Props):
    origin: str
    extent: gws.Extent
    resolutions: list[float]
    tileSize: int



