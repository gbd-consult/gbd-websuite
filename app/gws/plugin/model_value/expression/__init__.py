"""Expression value.

Computes the value by evaluating a Python expression. The expression is
compiled at configuration time; a syntax error is a configuration error.

The following variables are available in the expression:

- ``app``: the Application object,
- ``user``: the current user,
- ``project``: the current project,
- ``feature``: the feature the value is computed for,
- ``mc``: the ``gws.ModelContext`` object,
- ``date``: the ``gws.lib.datetimex`` module.

Additional modules can be made available with the ``imports`` option, a list
of module names, e.g. ``["math", "os.path"]``. Each module is available under
its top-level package name, so ``os.path`` is used as ``os.path.join(...)``.
If a module cannot be imported or the evaluation fails, the error is logged
and the value is None.

Example::

    fields+ {
        name "area_ha"
        type "float"
        values+ {
            type "expression"
            expression "round(feature.get('area_m2', 0) / 10000, 2)"
        }
    }
"""

from typing import Optional
import gws
import gws.base.model.value
import gws.lib.datetimex


@gws.ext.config.modelValue('expression')
class Config(gws.base.model.value.Config):
    """Value computed by a Python expression."""

    expression: str
    """Python expression to evaluate."""

    imports: Optional[list[str]]
    """Additional Python modules available to the expression."""


@gws.ext.object.modelValue('expression')
class Object(gws.base.model.value.Object):
    """Expression value object."""

    expression: str
    """Python expression to evaluate."""
    imports: list[str]
    """Names of modules imported for the expression."""

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
                context[mod.split('.')[0]] = __import__(mod)
            except ImportError as exc:
                gws.log.error(f'failed to import module {mod!r}: {exc!r}')
                return

        try:
            return eval(self.expression, context)
        except Exception as exc:
            gws.log.error(f'failed to compute expression: {exc!r}')
