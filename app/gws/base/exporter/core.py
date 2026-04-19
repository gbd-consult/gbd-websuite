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

    title: str
    supportsVector: bool
    supportsRaster: bool

    def configure(self):
        self.title = self.cfg('title')
        self.targetType = self.cfg('targetType')
        self.targetPath = self.cfg('targetPath')

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
