"""Pure python templates.

A template is a python module that has a function `main`,
which receives a Data object with context vars
and should return a `gws.ContentResponse` object.
"""

import re

import gws
import gws.base.template
import gws.types as t


@gws.ext.Config('template.py')
class Config(gws.base.template.Config):
    """Python template"""
    pass


@gws.ext.Object('template.py')
class Object(gws.base.template.Object):
    def configure(self):
        self.compile()

    def render(self, tri, notify=None):
        fn = self.compile()

        ctx = self.prepare_context(tri.context)
        if isinstance(ctx, dict):
            ctx = gws.Data(ctx)

        try:
            return fn(ctx)
        except Exception as exc:
            gws.log.exception()
            raise gws.Error(f'py error: {exc!r} path={self.path!r}') from exc

    def compile(self):
        text = gws.read_file(self.path) if self.path else self.text
        try:
            g = {}
            exec(text, g)
            return g['main']
        except Exception as exc:
            raise gws.Error(f'py load error: {exc!r} in {self.path!r}') from exc
