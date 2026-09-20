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
    """Namespace version. (deprecated in 8.5)"""
    extendsGml: bool = True
    """Namespace schema extends the GML3 schema."""


class Config(gws.Config):
    """XML helper."""

    namespaces: list[NamespaceConfig]
    """List of custom namespaces for XML generation."""


class Object(gws.Node):
    def configure(self):
        for c in self.cfg('namespaces', default=[]):
            self.add_namespace(c)

    def add_namespace(self, cfg: NamespaceConfig) -> gws.XmlNamespace:
        """Add a custom namespace for XML generation and register it globally."""

        xmlns = cfg.get('xmlns')
        if cfg.get('version'):
            self.root.config_warning(f'"xml.namespaces.version" is deprecated and ignored (namespace {xmlns!r})')
        ns = gws.lib.xmlx.namespace.new(
            prefix=xmlns,
            uri=cfg.get('uri'),
            schemaLocation=cfg.get('schemaLocation') or '',
            extendsGml=cfg.get('extendsGml', True),
        )

        try:
            gws.lib.xmlx.namespace.register(ns)
        except gws.lib.xmlx.Error as exc:
            raise gws.ConfigurationError(str(exc)) from exc

        return ns
