import gws
import gws.types as t
from . import namespaces


class NamespaceConfig(gws.Config):
    """XML namespace configuration"""

    name: str  #: namespace name
    uri: gws.Url  #: namespace uri
    schemaLocation: t.Optional[gws.Url]  #: namespace schema location


@gws.ext.Config('helper.xml')
class Config(gws.Config):
    """XML settings"""

    namespaces: t.Optional[t.List[NamespaceConfig]]  #: custom namespaces


@gws.ext.Object('helper.xml')
class Object(gws.Node):
    def activate(self):
        for ns in self.var('namespaces', default=[]):
            namespaces.add(ns.name, ns.uri, ns.schemaLocation)
