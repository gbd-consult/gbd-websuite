"""CX templates for XML."""

import re

import gws
import gws.tools.mime
import gws.gis.feature
import gws.gis.render
import gws.tools.pdf
import gws.tools.vendor.chartreux as chartreux
import gws.tools.xml2
import gws.common.template

import gws.types as t


class Config(gws.common.template.Config):
    """XML template"""
    pass


class XMLRuntime(chartreux.Runtime):
    def __init__(self):
        super().__init__()
        self.tags = [[]]
        self.ns = []
        self.last_tag = None
        self.namespaces_node = None

    def push_tag(self, name):
        tag = [name]
        self.tags[-1].append(tag)
        self.tags.append(tag)

    def pop_tag(self):
        self.last_tag = self.tags.pop()

    def append_tag(self, val):
        tag = self.tags[-1]
        tag.append(val)

    def set_attr(self, name, val):
        tag = self.tags[-1]
        tag.append({name: val})

    def set_text(self, s):
        tag = self.tags[-1]
        tag.append(s)

    def add_namespaces_here(self):
        self.namespaces_node = self.tags[-1]

    def as_text(self, *vals):
        return ''.join(str(v) for v in vals if v is not None)


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

    def command_namespaces(self, compiler: chartreux.Compiler, arg):
        compiler.code.add(f"_RT.add_namespaces_here()")

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


class Object(gws.common.template.Object):

    def render(self, context, format=None):
        namespaces_node, last_tag = self._render_as_tag(context)

        if namespaces_node:
            nsdict = self._collect_namespaces(iter(last_tag), {})
            nsdict.update(context.get('local_namespaces', {}))
            self._insert_namespaces(namespaces_node, nsdict, context.get('all_namespaces', {}))

        if format == 'tag':
            return t.TemplateOutput(content=last_tag)

        xml = gws.tools.xml2.as_string(last_tag)
        if not xml.startswith('<?'):
            xml = '<?xml version="1.0" encoding="utf-8"?>' + xml

        return t.TemplateOutput(content=xml)

    def _render_as_tag(self, context):
        context = context or {}

        context['gws'] = {
            'version': gws.VERSION,
            'endpoint': gws.SERVER_ENDPOINT,
        }

        def err(e, path, line):
            gws.log.warn(f'TEMPLATE: {e.__class__.__name__}:{e} in {path}:{line}')

        text = self.text
        if self.path:
            with open(self.path, 'rt') as fp:
                text = fp.read()

        rt = XMLRuntime()

        chartreux.render(
            text,
            context,
            path=self.path or '<string>',
            error=err,
            strip=True,
            runtime=rt,
            commands=XMLCommands(),
        )

        return rt.namespaces_node, rt.last_tag

    # @TODO: move this to the compiler
    def _collect_namespaces(self, tag_iter, nsdict):
        for el in tag_iter:
            n = el.find(':')
            if n > 0:
                nsdict.setdefault(el[:n], False)
            break

        for el in tag_iter:
            if isinstance(el, (list, tuple)):
                self._collect_namespaces(iter(el), nsdict)
            elif isinstance(el, dict):
                for key in el:
                    n = key.find(':')
                    if n > 0:
                        nsdict.setdefault(key[:n], False)

        return nsdict

    def _insert_namespaces(self, target_node, nsdict, all_namespaces):
        atts = {}
        schemas = []

        for id, ns in sorted(nsdict.items()):
            ns = ns or all_namespaces.get(id)
            if not ns:
                gws.log.warn(f'unknown namespace {id!r}')
                continue
            if isinstance(ns, str):
                uri, schema = ns, None
            else:
                uri, schema = ns
            atts['xmlns:' + id] = uri
            if schema:
                schemas.append(uri)
                schemas.append(schema)

        if atts:
            target_node.append(atts)

        if schemas:
            if 'xsi' not in nsdict:
                target_node.append({'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})
            target_node.append({'xsi:schemaLocation': ' '.join(schemas)})
