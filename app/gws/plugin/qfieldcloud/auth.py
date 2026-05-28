"""qfieldcloud authorisation method."""

import gws
import gws.base.auth

gws.ext.new.authMethod('qfieldcloud')


class Config(gws.base.auth.method.Config):
    """QField Cloud authorisation options. (added in 8.3)"""


class Object(gws.base.auth.method.Object):
    """QField Cloud authorisation method."""

    def configure(self):
        self.uid = 'gws.plugin.qfieldcloud.auth'
