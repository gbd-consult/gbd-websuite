from typing import Optional


import gws


class PositionConfig(gws.Config):
    """Position correction for points."""

    offsetX: int
    """X-offset for points."""
    offsetY: int
    """Y-offset for points."""
    distance: int = 0
    """Radius for points repelling."""
    angle: int = 0
    """Angle for points repelling."""


class SourceConfig(gws.Config):
    """Configuration for a gek-online source."""

    url: gws.Url
    """Base URL for gek-online calls."""
    params: dict
    """Parameters for gek-online calls."""
    instance: str
    """Instance name for gek-online calls, used to create unique uids."""


class IndexConfig(gws.Config):
    """Configuration for the GekoS index."""

    sources: list[SourceConfig]
    """List of gek-online sources."""
    position: Optional[PositionConfig]
    """Position correction for points."""
    tableName: str
    """SQL table name for storing GekoS data."""
    crs: gws.CrsName
    """CRS for GekoS data."""
    crs: gws.CrsName
    """CRS for gekos data."""
    dbUid: Optional[str]
    """Database provider uid."""
    sources: list[SourceConfig]
    """Gek-online instance names."""
    position: Optional[PositionConfig]
    """Position correction for points."""
    tableName: str
    """Sql table name."""
