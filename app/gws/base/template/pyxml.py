"""CX templates for XML."""

import re

import gws
import gws.base.template
import gws.lib.date
import gws.lib.mime
import gws.lib.xml3
import gws.lib.os2
import gws.types as t


@gws.ext.Config('template.pyxml')
class Config(gws.base.template.Config):
    """XML template"""
    pass


@gws.ext.Object('template.pyxml')
class Object(gws.base.template.Object):
    file_age: int

    def __getstate__(self):
        if hasattr(self, '_func'):
            delattr(self, '_func')
        return vars(self)

    def configure(self):
        self.file_age = gws.lib.os2.file_age(self.path) if self.path else 0
        self.compile()

    def render(self, context: dict, args=None):
        el = self.render_to_element(context)
        xml = gws.lib.xml3.to_string(el, with_xml=True, with_xmlns=True, with_schemas=True)
        return gws.TemplateRenderOutput(content=xml, mime=gws.lib.mime.XML)

    def render_to_element(self, context) -> gws.XmlElement:
        fn = self.compile()

        ctx = self.prepare_context(context['ARGS'])
        if isinstance(ctx, dict):
            ctx = gws.Data(ctx)

        try:
            return fn(ctx)
        except Exception as exc:
            raise gws.Error(f'pyxml runtime error: {exc!r} path={self.path!r}') from exc

    def compile(self):
        code = gws.read_file(self.path) if self.path else self.text

        # fn_name = '_template_py_xml'
        # indent = '    '
        # code = '\n'.join(indent + s for s in code.split('\n'))
        # code = f'def {fn_name}(args,tag):\n{code}\n{indent}return main(args,tag)\n'

        try:
            g = {}
            exec(code, g)
            return g['main']
        except Exception as exc:
            raise gws.Error(f'pyxml load error: {exc!r} in {self.path!r}') from exc
