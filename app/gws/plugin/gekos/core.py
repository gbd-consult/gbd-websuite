from typing import Optional


import gws


class PositionConfig(gws.Config):
    """Correction of point positions in the GekoS index."""

    offsetX: int
    """Offset added to X coordinates of points."""
    offsetY: int
    """Offset added to Y coordinates of points."""
    distance: int = 0
    """Radius of the circle on which points with the same location are spread."""
    angle: int = 0
    """Angle step in degrees for spreading points with the same location."""


class SourceConfig(gws.Config):
    """Gek-online source configuration."""

    url: gws.Url
    """Base URL for gek-online calls."""
    params: dict
    """Query parameters for gek-online calls."""
    instance: str
    """Instance name for gek-online calls, used to create unique UIDs."""


class IndexConfig(gws.Config):
    """Index of GekoS records in a database table."""

    sources: list[SourceConfig]
    """gek-online sources to load records from."""
    position: Optional[PositionConfig]
    """Position correction for points."""
    tableName: str
    """Database table for the GekoS index."""
    crs: gws.CrsName
    """CRS for GekoS data."""
    crs: gws.CrsName
    """CRS for GekoS data."""
    dbUid: Optional[str]
    """UID of the database provider for the index table."""
    sources: list[SourceConfig]
    """gek-online sources to load records from."""
    position: Optional[PositionConfig]
    """Position correction for points."""
    tableName: str
    """Database table for the GekoS index."""
