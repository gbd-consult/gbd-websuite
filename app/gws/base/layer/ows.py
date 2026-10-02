"""Layer OWS binding."""

from typing import Optional

import gws
import gws.lib.xmlx
import gws.config.util


class Config(gws.Config):
    """OWS service settings for a layer."""

    allowedServices: Optional[list[str]]
    """UIDs of OWS services allowed to publish this layer."""
    deniedServices: Optional[list[str]]
    """UIDs of OWS services that must not publish this layer."""
    featureName: str = ''
    """Feature type name in WFS."""
    geometryName: str = ''
    """Name of the geometry element in WFS and GML output."""
    layerName: str = ''
    """Layer name in WMS and WMTS services."""
    xmlns: Optional[str]
    """XML namespace prefix for the layer's features."""
    models: Optional[list[gws.ext.config.model]]
    """Data models for OWS output."""


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
