"""XML namespaces and options manager."""

import gws
import gws.gis.crs
import gws.lib.xmlx

from . import core

DEFAULT_NS_URI = '/_/owsXml/namespace/{}'
DEFAULT_NS_SCHEMA_LOCATION = '/_/owsXml/namespace/{}.xsd'


class Config(gws.Config):
    """XML configuration. (added in 8.1)"""

    namespaces: list[core.NamespaceConfig]


class Object(gws.Node):
    namespaces: list[gws.XmlNamespace]

    def configure(self):
        self.namespaces = []
        for c in self.cfg('namespaces'):
            self.add_namespace(c)

    def add_namespace(self, cfg: core.NamespaceConfig) -> gws.XmlNamespace:
        """Add a custom namespace for XML generation."""

        xmlns = cfg.get('xmlns')
        ns = gws.XmlNamespace(
            xmlns=xmlns,
            uid=cfg.get('uid') or 'gws.base.helper.xml.' + xmlns,
            uri=cfg.get('uri'),
            schemaLocation=cfg.get('schemaLocation'),
            version=cfg.get('version') or '',
            extendsGml=cfg.get('extendsGml') or True,
        )

        gws.lib.xmlx.namespace.register(ns)
        self.namespaces.append(ns)

        return ns

    def activate(self):
        for ns in self.namespaces:
            gws.lib.xmlx.namespace.register(ns)
