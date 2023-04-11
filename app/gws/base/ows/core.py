import gws
import gws.gis.crs
import gws.gis.source
import gws.lib.net
import gws.lib.mime
import gws.gis.ows
import gws.types as t


class OperationConfig(gws.Config):
    formats: t.Optional[list[str]]
    postUrl: t.Optional[gws.Url]
    url: gws.Url
    verb: gws.OwsVerb


class ProviderConfig(gws.Config):
    capsCacheMaxAge: gws.Duration = '1d'
    """max cache age for capabilities documents"""
    forceCrs: t.Optional[gws.CrsName]
    """use this CRS for requests"""
    alwaysXY: bool = False
    """force XY orientation for lat/lon projections"""
    maxRequests: int = 0
    """max concurrent requests to this source"""
    operations: t.Optional[list[OperationConfig]]
    """override operations reported in capabilities"""
    url: gws.Url
    """service url"""


class Caps(gws.Data):
    metadata: gws.Metadata
    operations: list[gws.OwsOperation]
    sourceLayers: list[gws.SourceLayer]
    tileMatrixSets: list[gws.TileMatrixSet]
    version: str
