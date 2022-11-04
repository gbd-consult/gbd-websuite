import gws
import gws.types as t

from . import namespace


class NamespaceConfig(gws.Config):
    """XML namespace configuration"""

    name: str 
    """namespace name"""
    uri: gws.Url 
    """namespace uri"""
    schemaLocation: t.Optional[gws.Url] 
    """namespace schema location"""


@gws.ext.config.helper('xml')
class Config(gws.Config):
    """XML settings"""

    namespaces: t.Optional[t.List[NamespaceConfig]] 
    """custom namespaces"""


@gws.ext.object.helper('xml')
class Object(gws.Node):
    def activate(self):
        for ns in self.var('namespaces', default=[]):
            namespace.register(ns.name, ns.uri, ns.schemaLocation)
