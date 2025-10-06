"""XML helper."""

from typing import Optional
import gws
import gws.lib.xmlx

gws.ext.new.helper('xml')


class NamespaceConfig(gws.Config):
    """XML Namespace configuration."""

    xmlns: str
    """Default prefix for this Namespace."""
    uri: gws.Url
    """Namespace uri."""
    schemaLocation: Optional[gws.Url]
    """Namespace schema location."""
    version: str = ''
    """Namespace version."""
    extendsGml: bool = True
    """Namespace schema extends the GML3 schema."""


class Config(gws.Config):
    """XML helper."""

    namespaces: list[NamespaceConfig]
    """List of custom namespaces for XML generation."""


class Object(gws.Node):
    namespaces: list[gws.XmlNamespace]

    def configure(self):
        self.namespaces = []
        for c in self.cfg('namespaces', default=[]):
            self.add_namespace(c)

    def add_namespace(self, cfg: NamespaceConfig) -> gws.XmlNamespace:
        """Add a custom namespace for XML generation."""

        xmlns = cfg.get('xmlns')
        ns = gws.XmlNamespace(
            xmlns=xmlns,
            uid=cfg.get('uid') or xmlns,
            uri=cfg.get('uri'),
            schemaLocation=cfg.get('schemaLocation') or '',
            version=cfg.get('version') or '',
            extendsGml=cfg.get('extendsGml') or True,
        )

        gws.lib.xmlx.namespace.register(ns)
        self.namespaces.append(ns)

        return ns

    def activate(self):
        for ns in self.namespaces:
            gws.lib.xmlx.namespace.register(ns)
