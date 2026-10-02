"""XML helper."""

from typing import Optional
import gws
import gws.lib.xmlx


class NamespaceConfig(gws.Config):
    """Custom XML namespace for generated XML documents."""

    xmlns: str
    """Default prefix for this namespace."""
    uri: gws.Url
    """Namespace URI."""
    schemaLocation: Optional[gws.Url]
    """URL of the namespace XML schema."""
    version: str = ''
    """Ignored. (deprecated in 8.5)"""
    extendsGml: bool = True
    """Namespace schema extends the GML3 schema."""


@gws.ext.config.helper('xml')
class Config(gws.Config):
    """Custom XML namespaces for generated XML documents."""

    namespaces: list[NamespaceConfig]
    """Custom namespaces to register."""


@gws.ext.object.helper('xml')
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
