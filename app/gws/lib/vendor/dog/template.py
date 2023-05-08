import os
import json
import tempfile
import re
import hashlib

import jump

from . import util, markdown

OPTIONS = dict(
    comment_symbol='#%',
    command_symbol='%',
    inline_open_symbol='{%',
    inline_close_symbol='%}',
    echo_open_symbol='<%',
    echo_close_symbol='%>',
    echo_start_whitespace=True,
)

GENERATED_NODE = '__DG__'


def compile(builder, path):
    try:
        return Engine(builder).compile_path(path, **OPTIONS)
    except jump.CompileError as exc:
        util.log.error(f'template compilation error: {exc.args[0]}')


def render(builder, text, path, args):
    try:
        return Engine(builder).render(text, args, error=_error, path=path, **OPTIONS)
    except jump.CompileError as exc:
        util.log.error(f'template compilation error: {exc.args[0]}')


def call(builder, tpl, args):
    return Engine(builder).call(tpl, args)


##


class Engine(jump.Engine):
    def __init__(self, builder):
        self.b = builder

    def generated_node(self, cls, args):
        args['class'] = cls
        js = json.dumps(args)
        return f'\n```\n{GENERATED_NODE}{js}\n```\n'

    def render_dot(self, text):
        tmp = os.path.join(tempfile.gettempdir(), util.random_string(8) + '.dot')
        util.write_file(tmp, text)
        ok, out = util.run(['dot', '-Tsvg', tmp], pipe=True)
        if not ok:
            return f'<xmp>DOT ERROR: {out}</xmp>'
        os.unlink(tmp)
        return '<svg' + out.split('<svg')[1]

    def wrap_html(self, before, text, after):
        return (
                self.generated_node('RawHtmlNode', {'html': before})
                + _dedent(text)
                + self.generated_node('RawHtmlNode', {'html': after})
        )

    def box_info(self, text):
        return self.wrap_html('<div class="admonition_info">', text, '</div>')

    def box_warn(self, text):
        return self.wrap_html('<div class="admonition_warn">', text, '</div>')

    def box_toc(self, text, depth=1):
        items = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        return self.generated_node('TocNode', {'items': items, 'depth': depth})

    def box_graph(self, text, caption=''):
        def go():
            svg = self.render_dot(text)
            cap = ''
            if caption:
                cap = '<figcaption>' + markdown.escape(caption) + '</figcaption>'
            return f'<figure>{svg}{cap}</figure>\n'

        return self.b.cached(_hash(text + caption), go)

    DBGRAPH_COLORS = {
        'text': '#455a64',
        'arrow': '#b0bec5',
        'head': '#1565c0',
        'border': '#b0bec5',
        'pk': '#ff8f00',
        'fk': '#00c5cf',
    }

    def box_dbgraph(self, text, caption=''):
        def span(s, color_name):
            c = self.DBGRAPH_COLORS[color_name]
            return f'<FONT COLOR="{c}">{s}</FONT>'

        def bold(s, color_name):
            c = self.DBGRAPH_COLORS[color_name]
            return f'<FONT COLOR="{c}"><B>{s}</B></FONT>'

        def parse_row(r):
            row = r.strip().split()
            k = row.pop() if row[-1].lower() in {'pk', 'fk'} else ''
            return [row[0], ' '.join(row[1:]) or ' ', k.lower()]

        def format_row(row, w0, w1):
            s = ''
            s += span(row[0], 'text') + ' ' * (w0 - len(row[0]))
            s += bold(row[1], 'text') + ' ' * (w1 - len(row[1]))
            if row[2] == 'pk':
                s += bold('&#x26bf;', 'pk')
            if row[2] == 'fk':
                s += bold('&#x26bf;', 'fk')
            return f'<TR><TD ALIGN="left" PORT="{row[0]}">{s}</TD></TR>'

        def make_table(name, body):
            rows = [parse_row(r) for r in body.strip().strip(',').split(',')]
            w0 = 2 + max(len(row[0]) for row in rows)
            w1 = 2 + max(len(row[1]) for row in rows)
            tbody = ''.join(format_row(row, w0, w1) for row in rows)
            thead = bold(name, 'head')
            c = self.DBGRAPH_COLORS['border']
            return f"""{name} [ label=<
                <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" COLOR="{c}">
                    <TR><TD>{thead}</TD></TR>
                    {tbody}
                </TABLE>
            >]"""

        def make_arrow(src, arr, dst):
            src = src.replace('.', ':')
            dst = dst.replace('.', ':')

            head = tail = "none"
            if arr == '>-':
                tail = 'crow'
            if arr == '-<':
                head = 'crow'

            c = self.DBGRAPH_COLORS['arrow']
            return f"""
                {src} -> {dst} [color="{c}", dir="both", arrowhead="{head}", arrowtail="{tail}"]    
            """

        def go():
            tables = ''.join(
                make_table(name, body)
                for name, body in re.findall(r'(?sx) (\w+) \s* \( (.+?) \)', text))

            arrows = ''.join(
                make_arrow(src, arr, dst)
                for src, arr, dst in re.findall(r'(?sx) ([\w.]+) \s* (>-|-<|--) \s* ([\w.]+)', text))

            dot = f"""
                digraph {{
                    rankdir="LR"
                    bgcolor="transparent"
                    splines="spline"
                    node [fontname="Menlo, monospace", fontsize=9, shape="plaintext"]
                    {tables}
                    {arrows}
                }}
            """

            return self.box_graph(dot, caption)

        return self.b.cached(_hash(text + caption), go)


##


def _error(exc, source_path, source_lineno, env):
    util.log.error(f'template error: {exc.args[0]} in {source_path!r}:{source_lineno}')


def _dedent(text):
    lines = text.split('\n')
    ind = 1e20
    for ln in lines:
        n = len(ln.lstrip())
        if n > 0:
            ind = min(ind, len(ln) - n)
    return '\n'.join(ln[ind:] for ln in lines)


def _hash(s):
    return hashlib.sha256(s.encode('utf8')).hexdigest()
