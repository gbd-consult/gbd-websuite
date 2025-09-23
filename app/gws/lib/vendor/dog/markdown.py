import re
from typing import List

import mistune
from mistune import Markdown

import pygments
import pygments.util
import pygments.lexers
import pygments.formatters.html

from . import util


class Element(util.Data):
    type: str

    align: str
    alt: str
    children: List['Element']
    info: str
    is_head: bool
    level: int
    target: str
    ordered: bool
    sid: str
    src: str
    start: str
    text: str
    html: str
    title: str

    classname: str  # inline_decoration_plugin
    attributes: dict  # link_attributes_plugin

    def __repr__(self):
        return repr(vars(self))


def parser() -> Markdown:
    md = mistune.create_markdown(renderer=AstRenderer(), plugins=['table', 'url', inline_decoration_plugin, link_attributes_plugin])
    return md


# plugin API reference: https://mistune.lepture.com/en/v2.0.5/advanced.html#create-plugins

# plugin: inline decorations
# {someclass some text} => <span class="decoration_someclass">some text</span>


def inline_decoration_plugin(md):
    name = 'inline_decoration'
    pattern = r'\{(\w+ .+?)\}'

    def parser(inline, m, state):
        return name, *m.group(1).split(None, 1)

    md.inline.register_rule(name, pattern, parser)
    md.inline.rules.append(name)


# plugin: link attributes
# https://pandoc.org/MANUAL.html#extension-link_attributes


def link_attributes_plugin(md):
    name = 'link_attributes'
    pattern = r'(?<=[)`]){.+?}'

    def parser(inline, m, state):
        text = m.group(0)
        atts = parse_attributes(text[1:-1])
        if atts:
            return name, text, atts
        return 'text', text

    md.inline.register_rule(name, pattern, parser)
    md.inline.rules.append(name)


##


def process(text):
    md = parser()
    els = md(text)
    rd = HTMLRenderer()
    return ''.join(rd.render_element(el) for el in els)


def strip_text_content(el: Element):
    while el.children:
        if not el.children[-1].text:
            return
        el.children[-1].text = el.children[-1].text.rstrip()
        if len(el.children[-1].text) > 0:
            return
        el.children.pop()


def text_from_element(el: Element) -> str:
    if el.text:
        return el.text.strip()
    if el.children:
        return ' '.join(text_from_element(c) for c in el.children).strip()
    return ''


# based on mistune/renderers.AstRenderer


class AstRenderer:
    NAME = 'ast'

    def __init__(self):
        self.parser = Parser()

    def register(self, name, method):
        pass

    def _get_method(self, name):
        return getattr(self.parser, f'p_{name}')

    def finalize(self, elements: List[Element]):
        # merge 'link attributes' with the previous element
        res = []
        for el in elements:
            if el.type == 'link_attributes':
                if res and res[-1].type in {'image', 'link', 'codespan'}:
                    res[-1].attributes = el.attributes
                    continue
                else:
                    el.type = 'text'
            res.append(el)
        return res


##


class Parser:
    def p_block_code(self, text, info=None):
        return Element(type='block_code', text=text, info=info)

    def p_block_error(self, children=None):
        return Element(type='block_error', children=children)

    def p_block_html(self, html):
        return Element(type='block_html', html=html)

    def p_block_quote(self, children=None):
        return Element(type='block_quote', children=children)

    def p_block_text(self, children=None):
        return Element(type='block_text', children=children)

    def p_codespan(self, text):
        return Element(type='codespan', text=text)

    def p_emphasis(self, children):
        return Element(type='emphasis', children=children)

    def p_heading(self, children, level):
        return Element(type='heading', children=children, level=level)

    def p_image(self, src, alt='', title=None):
        return Element(type='image', src=src, alt=alt, title=title)

    def p_inline_decoration(self, classname, text):
        return Element(type='inline_decoration', classname=classname, text=text)

    def p_inline_html(self, html):
        return Element(type='inline_html', html=html)

    def p_linebreak(self):
        return Element(type='linebreak')

    def p_link(self, target, children=None, title=None):
        if isinstance(children, str):
            children = [Element(type='text', text=children)]
        return Element(type='link', target=target, children=children, title=title)

    def p_link_attributes(self, text, attributes):
        return Element(type='link_attributes', text=text, attributes=attributes)

    def p_list_item(self, children, level):
        return Element(type='list_item', children=children, level=level)

    def p_list(self, children, ordered, level, start=None):
        return Element(type='list', children=children, ordered=ordered, level=level, start=start)

    def p_newline(self):
        return Element(type='newline')

    def p_paragraph(self, children=None):
        return Element(type='paragraph', children=children)

    def p_strong(self, children=None):
        return Element(type='strong', children=children)

    def p_table_body(self, children=None):
        return Element(type='table_body', children=children)

    def p_table_cell(self, children, align=None, is_head=False):
        return Element(type='table_cell', children=children, align=align, is_head=is_head)

    def p_table_head(self, children=None):
        return Element(type='table_head', children=children)

    def p_table(self, children=None):
        return Element(type='table', children=children)

    def p_table_row(self, children=None):
        return Element(type='table_row', children=children)

    def p_text(self, text):
        return Element(type='text', text=text)

    def p_thematic_break(self):
        return Element(type='thematic_break')


class _Renderer:
    def render_children(self, el: Element):
        if el.children:
            return ''.join(self.render_element(c) for c in el.children)
        return ''

    def render_element(self, el: Element):
        fn = getattr(self, f'r_{el.type}')
        return fn(el)


class MarkdownRenderer(_Renderer):
    def render_link(self, href, title, content, el):
        title = f' "{title}"' if title else ''
        return f'[{content}]({el.target}{title})'        

    def r_block_code(self, el: Element):
        lang = ''
        if el.info:
            lang = el.info.split(None, 1)[0]
        return f'```{lang}\n{el.text}\n```\n'

    def r_block_error(self, el: Element):
        c = self.render_children(el)
        return f'> **ERROR:** {c}\n\n'

    def r_block_html(self, el: Element):
        return el.html + '\n\n'

    def r_block_quote(self, el: Element):
        c = self.render_children(el)
        lines = c.split('\n')
        return ''.join(f'> {line}\n' for line in lines) + '\n'

    def r_block_text(self, el: Element):
        return self.render_children(el)

    def r_codespan(self, el: Element):
        return f'`{el.text}`'

    def r_emphasis(self, el: Element):
        return f'*{self.render_children(el)}*'

    def r_heading(self, el: Element):
        c = self.render_children(el)
        return f'{"#" * el.level} {c}\n\n'

    def r_image(self, el: Element):
        title = f' "{el.title}"' if el.title else ''
        return f'![{el.alt or ""}]({el.src}{title})'

    def r_inline_decoration(self, el: Element):
        return f'{{{el.classname} {el.text}}}'

    def r_inline_html(self, el: Element):
        return el.html

    def r_linebreak(self, el: Element):
        return '\n'

    def r_link(self, el: Element):
        c = self.render_children(el)
        return self.render_link(el.target, el.title, c or el.target, el)

    def r_list_item(self, el: Element):
        c = self.render_children(el)
        indent = '  ' * (el.level - 1)
        marker = '1. ' if getattr(el, 'ordered', False) else '- '
        return f'{indent}{marker}{c}\n'

    def r_list(self, el: Element):
        c = self.render_children(el)
        return c + '\n'

    def r_newline(self, el: Element):
        return '\n'

    def r_paragraph(self, el: Element):
        c = self.render_children(el)
        return f'{c}\n\n'

    def r_strong(self, el: Element):
        return f'**{self.render_children(el)}**'

    def r_table(self, el: Element):
        return self.render_children(el) + '\n'

    def r_table_head(self, el: Element):
        cells = [child for child in el.children if child.type == 'table_cell']
        header = '| ' + ' | '.join(self.render_children(cell) for cell in cells) + ' |\n'

        # Create the separator row based on alignment
        separators = []
        for cell in cells:
            if cell.align == 'center':
                separators.append(':---:')
            elif cell.align == 'right':
                separators.append('---:')
            else:  # left or None
                separators.append('---')

        separator = '| ' + ' | '.join(separators) + ' |\n'
        return header + separator

    def r_table_body(self, el: Element):
        return self.render_children(el)

    def r_table_row(self, el: Element):
        cells = [child for child in el.children if child.type == 'table_cell']
        return '| ' + ' | '.join(self.render_children(cell) for cell in cells) + ' |\n'

    def r_table_cell(self, el: Element):
        return self.render_children(el)

    def r_text(self, el: Element):
        return el.text

    def r_thematic_break(self, el: Element):
        return '---\n\n'


class HTMLRenderer(_Renderer):
    def render_link(self, href, title, content, el):
        a = {'href': href}
        if title:
            a['title'] = escape(title)
        if el.attributes:
            a.update(el.attributes)
        return f'<a{attributes(a)}>{content or href}</a>'

    ##

    def r_block_code(self, el: Element):
        lang = ''
        atts = {}

        lines = [s.rstrip() for s in el.text.split('\n')]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        text = '\n'.join(lines)

        if el.info:
            # 'javascript' or 'javascript title=...' or 'title=...'
            m = re.match(r'^(\w+(?=(\s|$)))?(.*)$', el.info.strip())
            if m:
                lang = m.group(1)
                atts = parse_attributes(m.group(3))

        lang = lang or 'text'
        try:
            lexer = pygments.lexers.get_lexer_by_name(lang, stripall=True)
        except pygments.util.ClassNotFound:
            util.log.warning(f'pygments lexer {lang!r} not found')
            lexer = pygments.lexers.get_lexer_by_name('text', stripall=True)

        kwargs = dict(
            noclasses=True,
            nobackground=True,
        )
        if 'numbers' in atts:
            kwargs['linenos'] = 'table'
            kwargs['linenostart'] = atts['numbers']

        formatter = pygments.formatters.html.HtmlFormatter(**kwargs)
        html = pygments.highlight(text, lexer, formatter)

        if 'title' in atts:
            html = f'<p class="highlighttitle">{escape(atts["title"])}</p>' + html

        return html

    def r_block_error(self, el: Element):
        c = self.render_children(el)
        return f'<div class="error">{c}</div>\n'

    def r_block_html(self, el: Element):
        return el.html

    def r_block_quote(self, el: Element):
        c = self.render_children(el)
        return f'<blockquote>\n{c}</blockquote>\n'

    def r_block_text(self, el: Element):
        return self.render_children(el)

    def r_codespan(self, el: Element):
        c = escape(el.text)
        return f'<code{attributes(el.attributes)}>{c}</code>'

    def r_emphasis(self, el: Element):
        c = self.render_children(el)
        return f'<em>{c}</em>'

    def r_heading(self, el: Element):
        c = self.render_children(el)
        tag = 'h' + str(el.level)
        s = ''
        if el.id:
            s += f' id="{el.id}"'
        return f'<{tag}{s}>{c}</{tag}>\n'

    def r_image(self, el: Element):
        a = {}
        if el.src:
            a['src'] = el.src
        if el.alt:
            a['alt'] = escape(el.alt)
        if el.title:
            a['title'] = escape(el.title)
        if el.attributes:
            a.update(el.attributes)
            n = a.pop('width', '')
            if n:
                if n.isdigit():
                    n += 'px'
                a['style'] = f'width:{n};' + a.get('style', '')
            n = a.pop('height', '')
            if n:
                if n.isdigit():
                    n += 'px'
                a['style'] = f'height:{n};' + a.get('style', '')

        return f'<img{attributes(a)}/>'

    def r_inline_decoration(self, el: Element):
        c = escape(el.text)
        return f'<span class="decoration_{el.classname}">{c}</span>'

    def r_inline_html(self, el: Element):
        return el.html

    def r_linebreak(self, el: Element):
        return '<br/>\n'

    def r_link(self, el: Element):
        c = self.render_children(el)
        return self.render_link(el.target, el.title, c, el)

    def r_list_item(self, el: Element):
        c = self.render_children(el)
        return f'<li>{c}</li>\n'

    def r_list(self, el: Element):
        c = self.render_children(el)
        tag = 'ol' if el.ordered else 'ul'
        a = {}
        if el.start:
            a['start'] = el.start
        return f'<{tag}{attributes(a)}>\n{c}\n</{tag}>\n'

    def r_newline(self, el: Element):
        return ''

    def r_paragraph(self, el: Element):
        c = self.render_children(el)
        return f'<p>{c}</p>\n'

    def r_strong(self, el: Element):
        c = self.render_children(el)
        return f'<strong>{c}</strong>'

    def r_table_body(self, el: Element):
        c = self.render_children(el)
        return f'<tbody>\n{c}</tbody>\n'

    def r_table_cell(self, el: Element):
        c = self.render_children(el)
        tag = 'th' if el.is_head else 'td'
        a = {}
        if el.align:
            a['style'] = f'text-align:{el.align}'
        return f'<{tag}{attributes(a)}>{c}</{tag}>'

    def r_table_head(self, el: Element):
        c = self.render_children(el)
        return f'<thead>\n<tr>{c}</tr>\n</thead>\n'

    def r_table(self, el: Element):
        c = self.render_children(el)
        return f'<table class="markdown-table">{c}</table>\n'

    def r_table_row(self, el: Element):
        c = self.render_children(el)
        return f'<tr>{c}</tr>\n'

    def r_text(self, el: Element):
        return escape(el.text)

    def r_thematic_break(self, el: Element):
        return '<hr/>\n'


def escape(s, quote=True):
    s = s.replace('&', '&amp;')
    s = s.replace('<', '&lt;')
    s = s.replace('>', '&gt;')
    if quote:
        s = s.replace('"', '&quot;')
    return s


def attributes(attrs):
    s = ''
    if attrs:
        for k, v in attrs.items():
            s += f' {k}="{v}"'
    return s


##


_ATTRIBUTE_RE = r"""(?x)
    (
        (\# (?P<id> [\w-]+) )
        |
        (\. (?P<class> [\w-]+) )
        |
        (
            (?P<key> \w+)
            =
            (
                " (?P<quoted> [^"]*) "
                |
                (?P<simple> \S+)
            )
        )
    )
    \x20
"""


def parse_attributes(text):
    text = text.strip() + ' '
    res = {}

    while text:
        m = re.match(_ATTRIBUTE_RE, text)
        if not m:
            return {}

        text = text[m.end() :].lstrip()

        g = m.groupdict()
        if g['id']:
            res['id'] = g['id']
        elif g['class']:
            res['class'] = (res.get('class', '') + ' ' + g['class']).strip()
        else:
            res[g['key']] = g['simple'] or g['quoted'].strip()

    return res
