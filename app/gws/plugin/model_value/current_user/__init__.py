"""Current user value.

Computes a string from the properties of the current user.

If ``format`` is configured, it is a Python format string that refers to the
``user`` object, e.g. ``"{user.displayName}_{user.isGuest}"``. Otherwise the
user's ``loginName`` is returned.

Example::

    fields+ {
        name "updated_by"
        type "text"
        values+ {
            type "currentUser"
            format "{user.displayName}"
            forRead false
            forUpdate true
        }
    }
"""

import gws
import gws.base.model.value


@gws.ext.config.modelValue('currentUser')
class Config(gws.base.model.value.Config):
    """Value derived from the current user."""

    format: str = ''
    """Format string for the user value."""


@gws.ext.object.modelValue('currentUser')
class Object(gws.base.model.value.Object):
    """Current user value object."""

    format: str
    """Format string, empty to use the login name."""

    def configure(self):
        self.format = self.cfg('format', default='')

    def compute(self, field, feature, mc):
        if self.format:
            return self.format.format(user=mc.user)
        return mc.user.loginName
