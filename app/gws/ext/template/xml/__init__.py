"""CX templates for XML."""

import re

import gws
import gws.tools.mime
import gws.gis.feature
import gws.gis.render
import gws.tools.misc as misc
import gws.tools.pdf
import gws.types as t
import gws.tools.chartreux
import gws.tools.chartreux.runtime
import gws.tools.xml3
import gws.common.template


class Config(t.TemplateConfig):
    """XML template"""
    pass


class XMLRuntime(gws.tools.chartreux.runtime.Runtime):
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


class XMLCommands():
    def __init__(self):
        self.tag_counts = []
        self.cc = None

    def interpolate(self, s):
        code = []

        for m, val in self.cc.command.parse_interpolations(s, with_default_filter=True):
            if m:
                code.append(f"str({val})")
            else:
                code.append(repr(val))
        return '+'.join(code)

    attr_re = r'''(?x)
        ([^\s=]+) = (
            " (?: .*?) "
            |
            \S+
        )
    '''

    def parse_atts(self, arg):
        while True:
            m = re.match(self.attr_re, arg)
            if not m:
                break
            a = self.interpolate(m.group(1))
            v = self.interpolate(m.group(2).strip('"').strip())
            self.cc.code.add(f"_RT.set_attr({a},{v})")
            arg = arg[len(m.group(0)):].strip()
        return arg

    def tag_header(self, arg):
        m = re.match(r'^(\S+)', arg)
        if not m:
            self.cc.error('invalid XML tag')

        cnt = 0
        for s in m.group(1).split('/'):
            s = self.interpolate(s)
            self.cc.code.add(f"_RT.push_tag({s})")
            cnt += 1
        self.tag_counts.append(cnt)

        arg = arg[len(m.group(0)):].strip()
        arg = self.parse_atts(arg)
        return arg.strip()

    def end_tag(self):
        cnt = self.tag_counts.pop()
        while cnt > 0:
            self.cc.code.add(f"_RT.pop_tag()")
            cnt -= 1

    def command_t(self, cc, arg):
        self.cc = cc
        arg = self.tag_header(arg)
        if arg:
            arg = self.interpolate(arg)
            self.cc.code.add(f"_RT.set_text({arg})")
        self.end_tag()

    def command_tag(self, cc, arg):
        self.cc = cc
        arg = self.tag_header(arg)
        if arg:
            self.cc.error('text is not allowed here')

        self.cc.code.add('_PUSHBUF()')
        self.cc.parser.parse_until('end')
        self.cc.code.add('_RT.set_text(_POPBUF())')
        self.end_tag()

    def command_a(self, cc, arg):
        self.cc = cc
        arg = self.parse_atts(arg)
        if arg:
            self.cc.error('text is not allowed here')

    def command_insert(self, cc, arg):
        self.cc = cc
        chunks = []
        for m, val in self.cc.command.parse_interpolations(arg, with_default_filter=False):
            if not m:
                self.cc.error('"insert" requires a single expression')
            chunks.append(val)

        if len(chunks) > 1:
            self.cc.error('"insert" requires a single expression')
        self.cc.code.add(f'_RT.append_tag({chunks[0]})')

    def command_namespaces(self, cc, arg):
        self.cc = cc
        self.cc.code.add(f"_RT.add_namespaces_here()")


class Object(gws.common.template.Object):

    @property
    def auto_uid(self):
        return None

    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.text = self.var('text')

        if self.path:
            fp = open(self.path, 'rt')
            fp.close()

        uid = self.var('uid') or (misc.sha256(self.path) if self.path else self.klass.replace('.', '_'))
        self.set_uid(uid)

    def render(self, context, render_output=None, out_path=None, format=None):
        namespaces_node, last_tag = self._render_as_tag(context)

        if namespaces_node:
            nsdict = self._collect_namespaces(iter(last_tag), {})
            nsdict.update(context.get('local_namespaces', {}))
            self._insert_namespaces(namespaces_node, nsdict, context.get('all_namespaces', {}))

        if format == 'tag':
            return t.Data({'content': last_tag})

        xml = gws.tools.xml3.as_string(last_tag)
        if not xml.startswith('<?'):
            xml = '<?xml version="1.0" encoding="utf-8"?>' + xml

        return t.Data({'content': xml})

    def _render_as_tag(self, context):
        context = context or {}

        context['gws'] = {
            'version': gws.VERSION,
            'endpoint': gws.SERVER_ENDPOINT,
        }

        def err(e, path, line):
            gws.log.warn(f'TEMPLATE: {e} at {path!r}:{line}')

        text = self.text
        if self.path:
            with open(self.path, 'rt') as fp:
                text = fp.read()

        rc = XMLCommands()
        rt = XMLRuntime()

        gws.tools.chartreux.render(
            text,
            context,
            path=self.path or '<string>',
            error=err,
            strip=True,
            runtime=rt,
            commands=rc,
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
