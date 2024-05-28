"""XML namespaces and options manager."""

from typing import cast

import gws
import gws.gis.crs
import gws.lib.xmlx

from . import core

DEFAULT_NS_URI = 'http://gbd-websuite.de/namespace/{}'
DEFAULT_NS_SCHEMA_LOCATION = '/_/owsXmlSchema/namespace/{}'


class Config(gws.Config):
    """XML configuration"""
    pass


class Object(gws.Node):
    namespaces: list[gws.XmlNamespace]

    def configure(self):
        self.namespaces = []

    def add_namespace(self, cfg: core.NamespaceConfig) -> gws.XmlNamespace:
        xmlns = cfg.get('xmlns')
        ns = gws.XmlNamespace(
            xmlns=xmlns,
            uid=cfg.get('uid') or 'gws.base.helper.xml.' + xmlns,
            uri=cfg.get('uri') or DEFAULT_NS_URI.format(xmlns),
            schemaLocation=cfg.get('schemaLocation') or DEFAULT_NS_SCHEMA_LOCATION.format(xmlns),
            version=cfg.get('version') or '',
            extendsGml=cfg.get('extendsGml') or True,
        )
        gws.lib.xmlx.namespace.register(ns)
        self.namespaces.append(ns)
        return ns

    def activate(self):
        for ns in self.namespaces:
            gws.lib.xmlx.namespace.register(ns)
