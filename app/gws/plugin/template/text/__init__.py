"""Text templates.

The ``text`` template is written in the Jump template language, like the
``html`` template, but has none of its custom commands (``@map``,
``@legend``, ``@page`` and so on) and produces text output only. The
output MIME type is the ``mimeOut`` of the render input, or the first
configured ``mimeTypes`` entry, or plain text. If a template explicitly
returns a :obj:`gws.Response` object, it is returned as the render result.

Example::

    templates+ {
        subject "feature.label"
        type "text"
        text "{{name}}"
    }
"""

from typing import Optional

import gws
import gws.base.legend
import gws.base.template
import gws.gis.render
import gws.lib.htmlx
import gws.lib.mime
import gws.lib.osx
import gws.lib.pdf
import gws.lib.vendor.jump


@gws.ext.config.template('text')
class Config(gws.base.template.Config):
    """Jump template for plain text output."""

    path: Optional[gws.FilePath]
    """Template file."""
    text: str = ''
    """Template source."""


@gws.ext.props.template('text')
class Props(gws.base.template.Props):
    pass


@gws.ext.object.template('text')
class Object(gws.base.template.Object):
    """Jump template for text output."""

    path: str
    """Template file path."""
    text: str
    """Template source."""
    compiledTime: float = 0
    """Time of the last compilation."""
    compiledFn = None
    """Compiled template function."""

    def configure(self):
        self.path = self.cfg('path')
        self.text = self.cfg('text', default='')
        if not self.path and not self.text:
            raise gws.Error('either "path" or "text" required')

    def render(self, tri):
        self.notify(tri, 'begin_print')

        engine = Engine()
        self.compile(engine)

        args = self.prepare_args(tri)
        res = engine.call(self.compiledFn, args=args, error=self.error_handler)

        if not isinstance(res, gws.Response):
            res = self.finalize(tri, res, args, engine)

        self.notify(tri, 'end_print')
        return res

    def compile(self, engine: 'Engine'):
        """Compile the template if needed.

        The template file is read again if it has changed since the last
        compilation. With the developer option ``template.always_reload``,
        the template is compiled on each call; with
        ``template.save_compiled``, the translated source is written to a
        debug file.

        Args:
            engine: Jump engine.
        """

        if self.path and (not self.text or gws.lib.osx.file_mtime(self.path) > self.compiledTime):
            self.text = gws.u.read_file(self.path)
            self.compiledFn = None

        if self.root.app.developer_option('template.always_reload'):
            self.compiledFn = None

        if not self.compiledFn:
            gws.log.debug(f'compiling {self} {self.path=}')
            if self.root.app.developer_option('template.save_compiled'):
                gws.u.write_debug_file(f'compiled_template_{self.uid}', engine.translate(self.text, path=self.path))

            self.compiledFn = engine.compile(self.text, path=self.path)
            self.compiledTime = gws.u.utime()

    def error_handler(self, exc, path, line, env):
        """Handle a template runtime error.

        The error is logged. With the developer option
        ``template.raise_errors``, the error is raised, otherwise rendering
        continues.

        Args:
            exc: The exception.
            path: Template path.
            line: Template line.
            env: Template environment.

        Returns:
            ``True`` to continue rendering, ``False`` to raise the error.
        """
        if self.root.app.developer_option('template.raise_errors'):
            gws.log.error(f'TEMPLATE_ERROR: {self}: {exc} IN {path}:{line}')
            return False

        gws.log.warning(f'TEMPLATE_ERROR: {self}: {exc} IN {path}:{line}')
        return True

    ##

    def finalize(self, tri: gws.TemplateRenderInput, res: str, args: dict, main_engine: 'Engine'):
        """Wrap the generated text in a content response.

        Args:
            tri: Template render input.
            res: Generated text.
            args: Template arguments.
            main_engine: Engine that rendered the text.

        Returns:
            Content response.
        """
        self.notify(tri, 'finalize_print')

        mime_type = tri.mimeOut
        if not mime_type and self.mimeTypes:
            mime_type = self.mimeTypes[0]
        if not mime_type:
            mime_type = gws.lib.mime.TXT

        return gws.ContentResponse(mimeType=mime_type, content=res)


##


class Engine(gws.lib.vendor.jump.Engine):
    """Jump engine for text templates, without custom commands."""

    pass
