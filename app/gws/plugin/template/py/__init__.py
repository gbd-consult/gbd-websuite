"""Python templates.

A ``py`` template is a Python module. The module must provide a function
called ``main``, which receives the template arguments object and returns
a :obj:`gws.Response` object. The module is loaded when the template is
configured and loaded again on each render, so changes to the file take
effect immediately.

Example::

    templates+ {
        subject "feature.description"
        type "py"
        path "/data/templates/description.py"
    }

with ``description.py``::

    import gws

    def main(args):
        return gws.ContentResponse(mimeType='text/plain', content=f'Subject: {args.subject}')
"""

from typing import Optional

import gws
import gws.base.template


@gws.ext.config.template('py')
class Config(gws.base.template.Config):
    """Template implemented as a Python module with a main function."""

    path: Optional[gws.FilePath]
    """Python module file with the main function."""


@gws.ext.props.template('py')
class Props(gws.base.template.Props):
    pass


_ENTRYPOINT_NAME = 'main'


@gws.ext.object.template('py')
class Object(gws.base.template.Object):
    """Template implemented as a Python module with a ``main`` function."""

    path: str
    """Python module file path."""

    def configure(self):
        self.path = self.cfg('path')
        self.compile()

    def render(self, tri):
        self.notify(tri, 'begin_print')

        args = self.prepare_args(tri)
        entrypoint = self.compile()

        try:
            res = entrypoint(args)
        except Exception as exc:
            # @TODO stack traces with the filename
            raise gws.Error(f'py error: {exc!r} path={self.path!r}') from exc

        self.notify(tri, 'end_print')
        return res

    def compile(self):
        """Load the module and return its ``main`` function.

        Returns:
            The ``main`` function.

        Raises:
            ``gws.Error``: If the module cannot be executed or has no ``main`` function.
        """
        text = gws.u.read_file(self.path)
        try:
            g = {}
            exec(text, g)
            return g[_ENTRYPOINT_NAME]
        except Exception as exc:
            raise gws.Error(f'py load error: {exc!r} in {self.path!r}') from exc
