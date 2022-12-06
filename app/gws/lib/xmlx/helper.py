"""XML helper object.

Used in the configuration to configure XML options and additional namespaces.
"""

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
        p = self.var('namespaces')
        if p:
            for ns in p:
                namespace.register(ns.name, ns.uri, schema=ns.schemaLocation)
