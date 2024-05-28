from typing import Optional
import gws


class NamespaceConfig(gws.Config):
    """XML Namespace configuration."""

    xmlns: str
    """Default prefix for this Namespace."""
    uri: Optional[gws.Url]
    """Namespace uri."""
    schemaLocation: Optional[gws.Url]
    """Namespace schema location."""
    version: str = ''
    """Namespace version."""
    extendsGml: bool = True
    """Namespace schema extends the GML3 schema."""
