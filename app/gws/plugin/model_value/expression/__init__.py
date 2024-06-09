"""Expression value.

This value is computed by evaluating a python expression.

The following variables are provided to the expression:

- ``app`` - the Application object
- ``user`` - current user
- ``feature`` - feature this value is evaluated for
- ``mc`` - ``ModelContext`` object

The following modules are available:

- ``date`` - `gws.lib.datetimex` module

"""

import gws
import gws.base.model.value
import gws.lib.datetimex

gws.ext.new.modelValue('expression')


class Config(gws.base.model.value.Config):
    text: str


class Object(gws.base.model.value.Object):
    text: str

    def configure(self):
        self.text = self.cfg('text')
        try:
            compile(self.text, 'expression', 'eval')
        except Exception as exc:
            raise gws.ConfigurationError(f'invalid expression: {exc!r}') from exc

    def compute(self, field, feature, mc):
        context = {
            'app': self.root.app,
            'user': mc.user,
            'feature': feature,
            'mc': mc,
            'date': gws.lib.datetimex
        }

        try:
            return eval(self.text, context)
        except Exception as exc:
            gws.log.error(f'failed to compute expression: {exc!r}')
