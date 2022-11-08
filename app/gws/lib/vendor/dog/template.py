import os
import json
import tempfile
import re

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

COMMAND = '__DG__'


def compile(path):
    try:
        return _ENGINE.compile_path(path, **OPTIONS)
    except jump.CompileError as exc:
        util.log.error(f'template compilation error: {exc.args[0]}')


def render(path, args):
    try:
        return _ENGINE.render_path(path, args, error=_error, **OPTIONS)
    except jump.CompileError as exc:
        util.log.error(f'template compilation error: {exc.args[0]}')


def call(tpl, args):
    return _ENGINE.call(tpl, args)


##


class Engine(jump.Engine):
    def emit_code_block(self, command, args):
        args['command'] = command
        js = json.dumps(args)
        return f'```\n{COMMAND}{js}\n```\n'

    def render_dot(self, text):
        tmp = os.path.join(tempfile.gettempdir(), util.random_string(8) + '.dot')
        util.write_file(tmp, text)
        ok, out = util.run(['dot', '-Tsvg', tmp], pipe=True)
        if not ok:
            return f'<xmp>DOT ERROR: {out}</xmp>'
        os.unlink(tmp)
        return '<svg' + out.split('<svg')[1]

    def box_toc(self, text, depth=1):
        items = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        return self.emit_code_block('TocNode', {'items': items, 'depth': depth})

    def box_graph(self, text, caption=''):
        svg = self.render_dot(text)

        cap = ''
        if caption:
            cap = '<figcaption>' + markdown.escape(caption) + '</figcaption>'

        return f'<figure>{svg}{cap}</figure>\n'

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

        def parse_row(row):
            c2 = row.pop() if row[-1].lower() in {'pk', 'fk'} else ' '
            c1 = row[1] if len(row) > 1 else ' '
            return [row[0], c1, c2]

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
            rows = [parse_row(s.strip().split()) for s in body.strip().split('\n')]
            w0 = 3 + max(len(row[0]) for row in rows)
            w1 = 3 + max(len(row[1]) for row in rows)
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
            mid = (src + '_' + dst).replace('.', '_')
            head = 'veevee' if arr == '->>' else 'vee'
            src = src.replace('.', ':')
            dst = dst.replace('.', ':')
            c = self.DBGRAPH_COLORS['arrow']
            return f"""
                {mid}          [shape=point width=0.001]
                {src} -> {mid} [color="{c}", arrowhead="{head}"]    
                {mid} -> {dst} [color="{c}", arrowhead="none"]    
            """

        tables = ''.join(
            make_table(name, body)
            for name, body in re.findall(r'(?sx) (\w+) \s* \( (.+?) \)', text))

        arrows = ''.join(
            make_arrow(src, arr, dst)
            for src, arr, dst in re.findall(r'(?sx) ([\w.]+) \s* (->>|->) \s* ([\w.]+)', text))

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


##

_ENGINE = Engine()


def _error(exc, source_path, source_lineno, env):
    util.log.error(f'template error: {exc.args[0]} in {source_path!r}:{source_lineno}')
