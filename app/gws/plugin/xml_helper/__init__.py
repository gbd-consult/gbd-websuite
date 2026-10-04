"""Helper for custom XML namespaces.

The helper registers custom XML namespaces, which can then be used in
generated XML documents. The namespaces are configured in the ``namespaces`` list and
registered globally in :obj:`gws.lib.xmlx.namespace` when the helper is
configured. Other code can add namespaces at run time with ``add_namespace``.

By default, a custom namespace is declared to extend the GML3 schema.

Example::

    helpers+ {
        type "xml"
        namespaces+ {
            xmlns "demo"
            uri "https://example.com/namespace/demo"
            schemaLocation "https://example.com/namespace/demo.xsd"
        }
    }

Example::

    helper = root.app.helper('xml')
    ns = helper.add_namespace(gws.Config(xmlns='demo', uri='https://example.com/namespace/demo'))
"""

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
    """XML helper, which registers custom namespaces."""

    def configure(self):
        for c in self.cfg('namespaces', default=[]):
            self.add_namespace(c)

    def add_namespace(self, cfg: NamespaceConfig) -> gws.XmlNamespace:
        """Create a custom namespace and register it globally.

        Args:
            cfg: Namespace configuration.

        Returns:
            The registered namespace.

        Raises:
            ``gws.ConfigurationError``: If the prefix or the URI conflicts with an already registered namespace.
        """

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
