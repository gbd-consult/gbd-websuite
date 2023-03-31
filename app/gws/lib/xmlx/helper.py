"""XML helper object.

Used in the configuration to configure XML options and additional namespaces.
"""

import gws
import gws.types as t

from . import namespace

gws.ext.new.helper('xml')


class NamespaceConfig(gws.Config):
    """XML namespace configuration"""

    name: str
    """namespace name"""
    uri: gws.Url
    """namespace uri"""
    schemaLocation: t.Optional[gws.Url]
    """namespace schema location"""


class Config(gws.Config):
    """XML settings"""

    namespaces: t.Optional[list[NamespaceConfig]]
    """custom namespaces"""


class Object(gws.Node):
    def activate(self):
        p = self.cfg('namespaces')
        if p:
            for ns in p:
                namespace.register(ns.name, ns.uri, schema=ns.schemaLocation)
