"""XML helper."""

import gws
import gws.lib.xmlx

gws.ext.new.helper('xml')


DEFAULT_NS_URI = '/_/owsXml/namespace/{}'
DEFAULT_NS_SCHEMA_LOCATION = '/_/owsXml/namespace/{}.xsd'


class NamespaceConfig(gws.Config):
    """XML Namespace configuration. (added in 8.1)"""

    xmlns: str
    """Default prefix for this Namespace."""
    uri: gws.Url
    """Namespace uri."""
    schemaLocation: gws.Url
    """Namespace schema location."""
    version: str = ''
    """Namespace version."""
    extendsGml: bool = True
    """Namespace schema extends the GML3 schema."""


class Config(gws.Config):
    """XML helper. (added in 8.1)"""

    namespaces: list[NamespaceConfig]


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
