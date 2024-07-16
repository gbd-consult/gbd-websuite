"""CX text-only templates.

This template is based on Jump, like html templates,
but doesn't support custom commands and non-text outputs.
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

gws.ext.new.template('text')


class Config(gws.base.template.Config):
    """Text-only template. (added in 8.1)"""

    path: Optional[gws.FilePath]
    """Path to a template file."""
    text: str = ''
    """Template content."""


class Props(gws.base.template.Props):
    pass


class Object(gws.base.template.Object):
    path: str
    text: str
    compiledTime: float = 0
    compiledFn = None

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

        if self.path and (not self.text or gws.lib.osx.file_mtime(self.path) > self.compiledTime):
            self.text = gws.u.read_file(self.path)
            self.compiledFn = None

        if self.root.app.developer_option('template.always_reload'):
            self.compiledFn = None

        if not self.compiledFn:
            gws.log.debug(f'compiling {self} {self.path=}')
            if self.root.app.developer_option('template.save_compiled'):
                gws.u.write_file(
                    gws.u.ensure_dir(f'{gws.c.VAR_DIR}/debug') + f'/compiled_template_{self.uid}',
                    engine.translate(self.text, path=self.path)
                )

            self.compiledFn = engine.compile(self.text, path=self.path)
            self.compiledTime = gws.u.utime()

    def error_handler(self, exc, path, line, env):
        if self.root.app.developer_option('template.raise_errors'):
            gws.log.error(f'TEMPLATE_ERROR: {self}: {exc} IN {path}:{line}')
            for k, v in sorted(getattr(env, 'ARGS', {}).items()):
                gws.log.error(f'TEMPLATE_ERROR: {self}: ARGS {k}={v!r}')
            gws.log.error(f'TEMPLATE_ERROR: {self}: stop')
            return False

        gws.log.warning(f'TEMPLATE_ERROR: {self}: {exc} IN {path}:{line}')
        return True

    ##

    def finalize(self, tri: gws.TemplateRenderInput, res: str, args: dict, main_engine: 'Engine'):
        self.notify(tri, 'finalize_print')

        mime = tri.mimeOut
        if not mime and self.mimeTypes:
            mime = self.mimeTypes[0]
        if not mime:
            mime = gws.lib.mime.TXT

        return gws.ContentResponse(mime=mime, content=res)


##


class Engine(gws.lib.vendor.jump.Engine):
    pass
