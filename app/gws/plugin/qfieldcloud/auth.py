"""Token authorization method for QField clients."""

import gws
import gws.base.auth


@gws.ext.config.authMethod('qfieldcloud')
class Config(gws.base.auth.method.Config):
    """Token authentication for QField clients."""


@gws.ext.object.authMethod('qfieldcloud')
class Object(gws.base.auth.method.Object):
    """QField Cloud authorization method.

    Token authorization for QField clients. The ``qfieldcloud`` action creates
    sessions with this method on login and accepts only tokens of its sessions.
    """

    def configure(self):
        self.uid = 'gws.plugin.qfieldcloud.auth'
