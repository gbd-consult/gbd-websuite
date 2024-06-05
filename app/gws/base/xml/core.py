import gws


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
