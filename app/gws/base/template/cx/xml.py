"""CX templates for XML."""

import re

import gws
import gws.types as t
import gws.base.template
import gws.lib.date
import gws.lib.mime
import gws.lib.vendor.chartreux as chartreux
import gws.lib.xml2
import gws.lib.xml2.helper




@gws.ext.Config('template.xml')
class Config(gws.base.template.Config):
    """XML template"""
    pass


class _InsertedTag:
    tag = None
    namespaces = None


class XMLRuntime(chartreux.Runtime):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.namespaces = {}
        self.default_namespace = None
        self.root = None

    def push_tag(self, name):
        self._add_ns(name)
        tag = [name]
        if self.tags:
            self.tags[-1].append(tag)
        else:
            self.root = tag
        self.tags.append(tag)

    def pop_tag(self):
        self.tags.pop()

    def append_tag(self, val):
        # "_InsertedTag" instance, maybe with namespaces
        if hasattr(val, 'tag'):
            if hasattr(val, 'namespaces'):
                self.namespaces.update(val.namespaces)
            val = val.tag

        self.tags[-1].append(val)

    def set_attr(self, name, val):
        self._add_ns(name)
        tag = self.tags[-1]
        tag.append({name: val})

    def set_text(self, s):
        tag = self.tags[-1]
        tag.append(s)

    def register_namespace(self, text):
        s = text.strip().split()
        if s:
            if s[-1] == 'default':
                self.default_namespace = s[0]
                s.pop()
            self.namespaces[s[0]] = s[1:]

    def as_text(self, *vals):
        return ''.join(str(v) for v in vals if v is not None)

    def _add_ns(self, name):
        if ':' in name:
            s = name.split(':')[0]
            if s not in self.namespaces:
                self.namespaces[s] = None

    def filter_datetime(self, val):
        d = gws.lib.date.parse(val)
        if d:
            return gws.lib.date.to_iso(d, with_tz=False, sep='T')
        return ''

    def filter_date(self, val):
        d = gws.lib.date.parse(val)
        if d:
            return gws.lib.date.to_iso_date(d)
        return ''


class XMLCommands():
    def __init__(self):
        self.tag_counts = []

    def command_t(self, compiler: chartreux.Compiler, arg):
        text = self.tag_header(compiler, arg)
        if text:
            text = self.interpolate(compiler, text)
            compiler.code.add(f"_RT.set_text({text})")
        self.end_tag(compiler)

    def command_tag(self, compiler: chartreux.Compiler, arg):
        text = self.tag_header(compiler, arg)
        if text:
            compiler.error('text is not allowed here')
        compiler.code.add('_PUSHBUF()')
        compiler.parser.parse_until('end')
        compiler.code.add('_RT.set_text(_POPBUF())')
        self.end_tag(compiler)

    def command_a(self, compiler: chartreux.Compiler, arg):
        text = self.parse_atts(compiler, arg)
        if text:
            compiler.error('text is not allowed here')

    def command_insert(self, compiler: chartreux.Compiler, arg):
        chunks = []
        for is_expr, val in compiler.command.parse_interpolations(arg, with_default_filter=False):
            if not is_expr:
                compiler.error('"insert" requires a single expression')
            chunks.append(val)
        if len(chunks) > 1:
            compiler.error('"insert" requires a single expression')
        compiler.code.add(f'_RT.append_tag({chunks[0]})')

    def command_xmlns(self, compiler: chartreux.Compiler, arg):
        # @xmlns wms default
        # @xmlns gml
        # @xmlns myNs http://url http://schema
        text = self.interpolate(compiler, arg)
        compiler.code.add(f"_RT.register_namespace({text})")

    ##

    def interpolate(self, compiler: chartreux.Compiler, s):
        vals = [
            v if is_expr else repr(v)
            for is_expr, v in compiler.command.parse_interpolations(s, with_default_filter=True)
        ]
        return '_RT.as_text(' + ','.join(vals) + ')'

    attr_re = r'''(?x)
        ([^\s=]+) = (
            " (?: .*?) "
            |
            \S+
        )
    '''

    def parse_atts(self, compiler: chartreux.Compiler, arg):
        while True:
            m = re.match(self.attr_re, arg)
            if not m:
                break
            a = self.interpolate(compiler, m.group(1))
            v = self.interpolate(compiler, m.group(2).strip('"').strip())
            compiler.code.add(f"_RT.set_attr({a},{v})")
            arg = arg[len(m.group(0)):].strip()
        return arg

    def tag_header(self, compiler: chartreux.Compiler, arg):
        m = re.match(r'^(\S+)', arg)
        if not m:
            compiler.error('invalid XML tag')
            return

        cnt = 0
        for s in m.group(1).split('/'):
            s = self.interpolate(compiler, s)
            compiler.code.add(f"_RT.push_tag({s})")
            cnt += 1
        self.tag_counts.append(cnt)

        arg = arg[len(m.group(0)):].strip()
        arg = self.parse_atts(compiler, arg)
        return arg.strip()

    def end_tag(self, compiler: chartreux.Compiler):
        cnt = self.tag_counts.pop()
        while cnt > 0:
            compiler.code.add(f"_RT.pop_tag()")
            cnt -= 1


@gws.ext.Object('template.xml')
class Object(gws.base.template.Object):
    helper: t.Optional[gws.lib.xml2.helper.Object]

    def configure(self):
        if self.path:
            self.text = gws.read_file(self.path)
        self.helper = t.cast(
            gws.lib.xml2.helper.Object,
            self.root.application.helper('xml'))

    def render(self, context: dict, mro=None, out_path=None, legends=None, format=None):
        rt = self._render_as_tag(context)
        root = rt.root

        if format == 'tag':
            ins_tag = _InsertedTag()
            ins_tag.tag = root
            if rt.namespaces:
                ins_tag.namespaces = rt.namespaces
            return gws.TemplateOutput(content=ins_tag)

        if rt.namespaces:
            self._insert_namespaces(root, rt.namespaces, rt.default_namespace)

        xml = gws.lib.xml2.as_string(root)
        if not xml.startswith('<?'):
            xml = '<?xml version="1.0" encoding="utf-8"?>' + xml

        return gws.TemplateOutput(content=xml, mime=gws.lib.mime.XML)

    def _render_as_tag(self, context):
        if self.root.application.developer_option('template.reparse') and self.path:
            self.text = gws.read_file(self.path)

        if self.root.application.developer_option('template.save_compiled'):
            gws.write_file(
                gws.VAR_DIR + '/debug_template_' + gws.as_uid(self.path),
                chartreux.translate(
                    self.text,
                    path=self.path or '<string>',
                    commands=XMLCommands()))

        def err(e, path, line):
            gws.log.error(f'TEMPLATE: {e.__class__.__name__}:{e} in {path}:{line}')

        if self.root.application.developer_option('template.raise_errors'):
            err = None

        rt = XMLRuntime()

        chartreux.render(
            self.text,
            self.prepare_context(context),
            path=self.path or '<string>',
            error=err,
            strip=True,
            runtime=rt,
            commands=XMLCommands(),
        )

        return rt

    def _insert_namespaces(self, target_node, nsdict, default_namespace):
        # a namespace can

        atts = {}
        schemas = []

        for id, ns in sorted(nsdict.items()):
            ns = ns or self.helper.namespaces.get(id)
            if not ns:
                gws.log.warn(f'unknown namespace {id!r}')
                continue

            if isinstance(ns, str):
                uri, schema = ns, None
            elif len(ns) == 1:
                uri, schema = ns[0], None
            else:
                uri, schema = ns

            atts['xmlns' if id == default_namespace else 'xmlns:' + id] = uri

            if schema:
                schemas.append(uri)
                schemas.append(schema)

        if schemas:
            atts['xmlns:xsi'] = self.helper.namespaces['xsi'][0]
            atts['xsi:schemaLocation'] = ' '.join(schemas)

        if atts:
            target_node.append(atts)
