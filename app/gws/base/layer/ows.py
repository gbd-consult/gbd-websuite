"""Layer OWS binding."""

from typing import Optional

import gws
import gws.lib.xmlx
import gws.config.util


class Config(gws.Config):
    """Layer OWS configuration."""

    allowedServices: Optional[list[str]]
    """Service UIDs which can use this layer."""
    deniedServices: Optional[list[str]]
    """Service UIDs which can not use this layer."""
    featureName: str = ''
    """Name for features in this layer."""
    geometryName: str = ''
    """Name for geometries in this layer."""
    layerName: str = ''
    """Name for this layer in WMS services."""
    xmlns: Optional[str]
    """XML namespace prefix."""
    models: Optional[list[gws.ext.config.model]]
    """OWS-specific models."""


class Object(gws.LayerOwsBinding):
    def configure(self):
        self.allowedServiceUids = self.cfg('allowedServices', default=[])
        self.deniedServiceUids = self.cfg('deniedServices', default=[])
        self.xmlNamespace = None

        p = self.cfg('xmlns')
        if p:
            self.xmlNamespace = self._namespace(p)

        self.layerName = self._configure_name('layerName') or self.cfg('_defaultName')
        self.featureName = self._configure_name('featureName') or self.cfg('_defaultName')
        # NB geometryName might come from a model later on
        self.geometryName = self._configure_name('geometryName') or ''

        gws.config.util.configure_models_for(self)

    def _configure_name(self, key):
        p = self.cfg(key)
        if not p:
            return
        _, prefix, pname = gws.lib.xmlx.namespace.parse_name(p)
        if prefix:
            self.xmlNamespace = self._namespace(prefix)
        return pname

    def _namespace(self, prefix):
        ns = gws.lib.xmlx.namespace.find_by_prefix(prefix)
        if not ns:
            raise gws.ConfigurationError(f'unknown XML namespace {prefix!r}')
        return ns
