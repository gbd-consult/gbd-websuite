"""Current user.

Formats properties of the current user and returns them as a string.

If ``format`` is configured, it must be a Python format string with a reference
to the ``user`` param, e.g. ``"{user.displayName}_{user.isGuest}"``.

If no ``format`` is configured, user's ``loginName`` is returned.

"""

import gws
import gws.base.model.value

gws.ext.new.modelValue('currentUser')


class Config(gws.base.model.value.Config):
    format: str = ''
    """format string"""


class Object(gws.base.model.value.Object):
    format: str

    def configure(self):
        self.format = self.cfg('format', default='')

    def compute(self, field, feature, mc):
        if self.format:
            return self.format.format(user=mc.user)
        return mc.user.loginName
