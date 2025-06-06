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
        def go():
            try:
                dot = _dbgraph_to_dot(text, self.DBGRAPH_COLORS)
            except Exception:
                return '<xmp>DBGRAPH SYNTAX ERROR</xmp>'
            return self.box_graph(dot, caption)

        return self.b.cached(_hash(text + caption), go)


##


def _dbgraph_to_dot(text, colors):
    nl = '\n'.join

    def span(s, color_name):
        c = colors[color_name]
        return f'<FONT COLOR="{c}">{s}</FONT>'

    def bold(s, color_name):
        c = colors[color_name]
        return f'<FONT COLOR="{c}"><B>{s}</B></FONT>'

    def make_table(rows, tab_name):
        w_col = 0
        tbody = []

        for tab, col, typ, pk, ref_tab, ref_col in rows:
            if tab != tab_name:
                continue
            w_col = max(w_col, len(col))

        for tab, col, typ, pk, ref_tab, ref_col in rows:
            if tab != tab_name:
                continue
            s = ''
            if pk:
                s += bold('&#x2B25;', 'pk')
            elif ref_tab:
                s += bold('&#x2B25;', 'fk')
            else:
                s += bold('&#x2B25;', 'arrow')
            s += ' ' 
            s += span((col or ' ').ljust(w_col), 'text')
            s += ' ' * 2
            s += bold(typ or ' ', 'text')

            tbody.append(f'<TR><TD ALIGN="left" PORT="{col}">{s}</TD></TR>')

        c = colors['border']

        return f"""{tab_name} [ label=<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" COLOR="{c}">
                <TR><TD> {bold(tab_name, 'head')} </TD></TR>
                {nl(tbody)}
            </TABLE>
        >]"""

    def make_arrow(tab, col, ref_tab, ref_col):
        src = f'{tab}:{col}'
        dst = f'{ref_tab}:{ref_col}'

        c = colors['arrow']
        d = 0.5

        return f"""
            {src} -> {dst} [
                color="{c}", 
                dir="both", 
                arrowtail="none", 
                arrowhead="vee", 
                arrowsize="{d}"
            ]
        """

    def parse(text):
        rows = []
        
        for m in re.findall(r'(?xs)(\w+) \s* \( (.*?) \)', text):
            tab, body = m
            for r in body.split(','):
                r = r.strip().split()
                if not r:
                    continue
                col = r.pop(0)
                typ, pk, ref_tab, ref_col = '', False, '', ''
                while r:
                    s = r.pop(0)
                    if s == 'pk':
                        pk = True
                    elif s == '->':
                        ref_tab, ref_col = r.pop(0).split('.')
                    else:
                        typ += s + ' '
                rows.append((tab, col, typ.strip(), pk, ref_tab, ref_col))

        return rows

    rows = parse(text)

    tables = []
    arrows = []

    for tab_name in set(r[0] for r in rows):
        tables.append(make_table(rows, tab_name))

    for tab, col, typ, pk, ref_tab, ref_col in rows:
        if ref_tab:
            arrows.append(make_arrow(tab, col, ref_tab, ref_col))

    return f"""
        digraph {{
            layout="dot"
            rankdir="LR"
            bgcolor="transparent"
            splines="spline"
            node [fontname="Menlo, monospace", fontsize=9, shape="plaintext"]
            {nl(tables)}
            {nl(arrows)}
        }}
    """


def _error(exc, source_path, source_lineno, env):
    util.log.error(f'template error: {exc.args[0]} in {source_path!r}:{source_lineno}')


def _dedent(text):
    lines = text.split('\n')
    ind = 100_000
    for ln in lines:
        n = len(ln.lstrip())
        if n > 0:
            ind = min(ind, len(ln) - n)
    return '\n'.join(ln[ind:] for ln in lines)


def _hash(s):
    return hashlib.sha256(s.encode('utf8')).hexdigest()
