"""Expression value.

This value is computed by evaluating a python expression.

The following variables are provided to the expression:

- ``app`` - the Application object
- ``user`` - current user
- ``project`` - current project
- ``feature`` - feature this value is evaluated for
- ``mc`` - ``ModelContext`` object

The following modules are available:

- ``date`` - `gws.lib.datetimex` module

Additional modules can be imported by specifying them in the
``imports`` configuration option. This should be a list of module names
to import, e.g. ``["math", "os"]``.

"""

from typing import Optional
import gws
import gws.base.model.value
import gws.lib.datetimex

gws.ext.new.modelValue('expression')


class Config(gws.base.model.value.Config):
    """Expression-based value. (added in 8.1)"""

    expression: str
    """Python expression to evaluate."""

    imports: Optional[list[str]]
    """List of additional modules to import. (added in 8.2)"""


class Object(gws.base.model.value.Object):
    expression: str
    imports: list[str]

    def configure(self):
        self.expression = (self.cfg('expression') or '').strip()
        self.imports = self.cfg('imports') or []
        try:
            compile(self.expression, 'expression', 'eval')
        except Exception as exc:
            raise gws.ConfigurationError(f'invalid expression: {exc!r}') from exc

    def compute(self, field, feature, mc):
        context = {
            'app': self.root.app,
            'user': mc.user,
            'project': mc.project,
            'feature': feature,
            'mc': mc,
            'date': gws.lib.datetimex,
        }
        for mod in self.imports:
            try:
                context[mod] = __import__(mod)
            except ImportError as exc:
                gws.log.error(f'failed to import module {mod!r}: {exc!r}')
                return

        try:
            return eval(self.expression, context)
        except Exception as exc:
            gws.log.error(f'failed to compute expression: {exc!r}')
