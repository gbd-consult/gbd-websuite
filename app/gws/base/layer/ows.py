"""Layer OWS controller."""

from typing import Optional

import gws
import gws.lib.xmlx
import gws.config.util


class Config(gws.Config):
    allowedServices: Optional[list[str]]
    deniedServices: Optional[list[str]]
    featureName: str = ''
    geometryName: str = ''
    layerName: str = ''
    xmlns: Optional[str]
    models: Optional[list[gws.ext.config.model]]


class Object(gws.LayerOws):
    def configure(self):
        self.allowedServiceUids = self.cfg('allowedServices', default=[])
        self.deniedServiceUids = self.cfg('deniedServices', default=[])
        self.xmlNamespace = None

        p = self.cfg('xmlns')
        if p:
            self.xmlNamespace, _ = gws.lib.xmlx.namespace.parse_name(p + ':test')

        self.layerName = self._configure_name('layerName') or self.cfg('_defaultName')
        self.featureName = self._configure_name('featureName') or self.cfg('_defaultName')
        # NB geometryName might come from a model later on
        self.geometryName = self._configure_name('geometryName') or ''

        gws.config.util.configure_models_for(self)

    def _configure_name(self, key):
        p = self.cfg(key)
        if not p:
            return
        ns, pname = gws.lib.xmlx.namespace.parse_name(p)
        if ns:
            self.xmlNamespace = ns
        return pname
