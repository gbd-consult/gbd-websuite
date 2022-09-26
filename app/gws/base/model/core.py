import gws
import gws.types as t


class Config(gws.Config):
    """Data model"""

    crs: t.Optional[gws.CrsName]  #: CRS for this model
    geometryType: t.Optional[gws.GeometryType]  #: specific geometry type


class RuleProps(gws.Props):
    editable: bool
    name: str
    title: str
    type: str


class Props(gws.Props):
    geometryType: str
    crs: str

@gws.ext.props.model('default')
class PropsD(gws.Props):
    geometryType: str
    crs: str

@gws.ext.props.model('postgres')
class PropsPostgres(gws.Props):
    geometryType: str
    crs: str



@gws.ext.object.model('default')
class Object(gws.Node, gws.IModel):
    pass