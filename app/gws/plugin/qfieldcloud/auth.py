"""qfieldcloud authorisation method."""

import gws
import gws.base.auth


@gws.ext.config.authMethod('qfieldcloud')
class Config(gws.base.auth.method.Config):
    """QField Cloud authorisation options."""


@gws.ext.object.authMethod('qfieldcloud')
class Object(gws.base.auth.method.Object):
    """QField Cloud authorisation method."""

    def configure(self):
        self.uid = 'gws.plugin.qfieldcloud.auth'
