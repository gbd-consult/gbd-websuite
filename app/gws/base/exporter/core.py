from typing import Optional

import gws


class Config(gws.ConfigWithAccess):
    """Exporter configuration."""

    title: str
    """Exporter title."""
    target: gws.ExportTarget
    """Target export type."""
    targetPath: Optional[str]
    """Target export path (if applicable)."""
    withNoGeometry: bool = False
    """Allow features with no geometry."""
    withMixedGeometry: bool = False
    """Allow features with different geometries in the same layer."""
    withMixedCrs: bool = False
    """Allow features with different CRS in the same layer."""
    withMultiLayer: bool = False
    """Store multiple layers in a single file."""
    options: dict = {}
    """Additional exporter-specific options."""


class Props(gws.Props):
    """Exporter properties."""

    title: str
    """Exporter title."""
    supportsVector: bool
    """Whether exporter supports vector features."""
    supportsRaster: bool
    """Whether exporter supports raster features."""


class Object(gws.Exporter):
    """Exporter object."""

    def configure(self):
        self.title = self.cfg('title')
        self.targetType = self.cfg('targetType')
        self.targetPath = self.cfg('targetPath')
        self.withNoGeometry = self.cfg('withNoGeometry')
        self.withMixedGeometry = self.cfg('withMixedGeometry')
        self.withMixedCrs = self.cfg('withMixedCrs')
        self.withMultiLayer = self.cfg('withMultiLayer') and self.supportsMultiLayer
        self.options = gws.u.to_dict(self.cfg('options')) or {}
        self.supportedAttributeTypes = []

    def props(self, user) -> Props:
        """Return exporter properties."""
        return Props(
            uid=self.uid,
            title=self.title,
            supportsVector=self.supportsVector,
            supportsRaster=self.supportsRaster,
        )

    def notify(self, ea: 'ExportArgs', event: str):
        """Notify exporter of export events."""
        if ea.notify:
            ea.notify(event)
