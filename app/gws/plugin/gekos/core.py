"""Interface with GekoS-Bau software.

See https://www.gekos.de/

GekoS settings for gws (Verfahrensadministration/GIS Schnittstelle)

base address:

    GIS-URL-Base  = http://my-server

client-side call, handled in the client by the Marker element

    GIS-URL-ShowXY  = /project/PROJECT_ID/?x=<x>&y=<y>&z=SCALE_VALUE

client-side call, handled in the client by js/index.tsx

    GIS-URL-GetXYFromMap = /project/PROJECT_ID/?&x=<x>&y=<y>&gekosUrl=<returl>

client-side call, handled in the Alkis plugin

    GIS-URL-ShowFs = /project/PROJECT_ID/?alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>

callback urls, handled by the GekoS action

    GIS-URL-GetXYFromFs   = /_/gekosGetXY/projectUid/PROJECT_ID/fs/<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
    GIS-URL-GetXYFromGrd  = /_/gekosGetXY/projectUid/PROJECT_ID/ad/<str>_<hnr><hnralpha>_<plz>_<ort>_<bishnr><bishnralpha>

NB: the order of placeholders must match COMBINED_FLURSTUECK_FIELDS and COMBINED_ADRESSE_FIELDS in the Alkis Plugin

"""

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
