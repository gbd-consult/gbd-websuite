import gws
import gws.base.feature
import gws.types as t


class SortConfig(gws.Config):
    fieldName: str
    order: str


class Config(gws.ConfigWithAccess):
    """Model configuration"""

    fields: t.List[gws.ext.config.modelField]
    sfields: t.List[gws.ext.config.template]
    filter: t.Optional[str]
    sort: t.Optional[t.List[SortConfig]]


class Props(gws.Props):
    geometryType: str
    crs: str


class Object(gws.Node, gws.IModel):
    UID_DELIMITER = '::'
