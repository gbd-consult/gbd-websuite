"""Pure python templates.

A template is a python module that has a function `main`,
which receives a dict with template arguments
and is supposed return a `gws.ContentResponse` object.
"""

import gws
import gws.base.template
import gws.types as t

gws.ext.new.template('py')


class Config(gws.base.template.Config):
    """Python template"""

    path: t.Optional[gws.FilePath]
    """path to a template file"""


class Props(gws.base.template.Props):
    pass


_ENTRYPOINT_NAME = 'main'


class Object(gws.base.template.Object):
    path: str

    def configure(self):
        self.path = self.cfg('path')
        self.compile()

    def render(self, tri):
        self.notify(tri, 'begin_print')

        args = self.prepare_args(tri.args)
        entrypoint = self.compile()

        try:
            res = entrypoint(args)
        except Exception as exc:
            # @TODO stack traces with the filename
            raise gws.Error(f'py error: {exc!r} path={self.path!r}') from exc

        self.notify(tri, 'end_print')
        return res

    def compile(self):
        text = gws.read_file(self.path)
        try:
            g = {}
            exec(text, g)
            return g[_ENTRYPOINT_NAME]
        except Exception as exc:
            raise gws.Error(f'py load error: {exc!r} in {self.path!r}') from exc
