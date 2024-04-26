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
    link: str
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
    md = mistune.create_markdown(
        renderer=AstRenderer(),
        plugins=['table', 'url', inline_decoration_plugin, link_attributes_plugin]
    )
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
    rd = Renderer()
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
        self.renderer = Renderer()

    def register(self, name, method):
        pass

    def _get_method(self, name):
        return getattr(self.renderer, f'{name}_parse')

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

class Renderer:

    def render_children(self, el: Element):
        if el.children:
            return ''.join(self.render_element(c) for c in el.children)
        return ''

    def render_element(self, el: Element):
        fn = getattr(self, f'{el.type}_render')
        return fn(el)

    def render_a(self, href, title, content, el):
        a = {'href': href}
        if title:
            a['title'] = escape(title)
        if el.attributes:
            a.update(el.attributes)
        return f'<a{attributes(a)}>{content or href}</a>'

    ##

    def block_code_parse(self, text, info=None):
        return Element(type='block_code', text=text, info=info)

    def block_code_render(self, el: Element):

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

    def block_error_parse(self, children=None):
        return Element(type='block_error', children=children)

    def block_error_render(self, el: Element):
        c = self.render_children(el)
        return f'<div class="error">{c}</div>\n'

    def block_html_parse(self, html):
        return Element(type='block_html', html=html)

    def block_html_render(self, el: Element):
        return el.html

    def block_quote_parse(self, children=None):
        return Element(type='block_quote', children=children)

    def block_quote_render(self, el: Element):
        c = self.render_children(el)
        return f'<blockquote>\n{c}</blockquote>\n'

    def block_text_parse(self, children=None):
        return Element(type='block_text', children=children)

    def block_text_render(self, el: Element):
        return self.render_children(el)

    def codespan_parse(self, text):
        return Element(type='codespan', text=text)

    def codespan_render(self, el: Element):
        c = escape(el.text)
        return f'<code{attributes(el.attributes)}>{c}</code>'

    def emphasis_parse(self, children):
        return Element(type='emphasis', children=children)

    def emphasis_render(self, el: Element):
        c = self.render_children(el)
        return f'<em>{c}</em>'

    def heading_parse(self, children, level):
        return Element(type='heading', children=children, level=level)

    def heading_render(self, el: Element):
        c = self.render_children(el)
        tag = 'h' + str(el.level)
        s = ''
        if el.id:
            s += f' id="{el.id}"'
        return f'<{tag}{s}>{c}</{tag}>\n'

    def image_parse(self, src, alt="", title=None):
        return Element(type='image', src=src, alt=alt, title=title)

    def image_render(self, el: Element):
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
                a['style'] = f"width:{n};" + a.get('style', '')
            n = a.pop('height', '')
            if n:
                if n.isdigit():
                    n += 'px'
                a['style'] = f"height:{n};" + a.get('style', '')

        return f'<img{attributes(a)}/>'

    def inline_decoration_parse(self, classname, text):
        return Element(type='inline_decoration', classname=classname, text=text)

    def inline_decoration_render(self, el: Element):
        c = escape(el.text)
        return f'<span class="decoration_{el.classname}">{c}</span>'

    def inline_html_parse(self, html):
        return Element(type='inline_html', html=html)

    def inline_html_render(self, el: Element):
        return el.html

    def linebreak_parse(self):
        return Element(type='linebreak')

    def linebreak_render(self, el: Element):
        return '<br/>\n'

    def link_parse(self, link, children=None, title=None):
        if isinstance(children, str):
            children = [Element(type='text', text=children)]
        return Element(type='link', link=link, children=children, title=title)

    def link_render(self, el: Element):
        c = self.render_children(el)
        return self.render_a(el.link, el.title, c, el)

    def link_attributes_parse(self, text, attributes):
        return Element(type='link_attributes', text=text, attributes=attributes)

    def list_item_parse(self, children, level):
        return Element(type='list_item', children=children, level=level)

    def list_item_render(self, el: Element):
        c = self.render_children(el)
        return f'<li>{c}</li>\n'

    def list_parse(self, children, ordered, level, start=None):
        return Element(type='list', children=children, ordered=ordered, level=level, start=start)

    def list_render(self, el: Element):
        c = self.render_children(el)
        tag = 'ol' if el.ordered else 'ul'
        a = {}
        if el.start:
            a['start'] = el.start
        return f'<{tag}{attributes(a)}>\n{c}\n</{tag}>\n'

    def newline_parse(self):
        return Element(type='newline')

    def newline_render(self, el: Element):
        return ''

    def paragraph_parse(self, children=None):
        return Element(type='paragraph', children=children)

    def paragraph_render(self, el: Element):
        c = self.render_children(el)
        return f'<p>{c}</p>\n'

    def strong_parse(self, children=None):
        return Element(type='strong', children=children)

    def strong_render(self, el: Element):
        c = self.render_children(el)
        return f'<strong>{c}</strong>'

    def table_body_parse(self, children=None):
        return Element(type='table_body', children=children)

    def table_body_render(self, el: Element):
        c = self.render_children(el)
        return f'<tbody>\n{c}</tbody>\n'

    def table_cell_parse(self, children, align=None, is_head=False):
        return Element(type='table_cell', children=children, align=align, is_head=is_head)

    def table_cell_render(self, el: Element):
        c = self.render_children(el)
        tag = 'th' if el.is_head else 'td'
        a = {}
        if el.align:
            a['style'] = f'text-align:{el.align}'
        return f'<{tag}{attributes(a)}>{c}</{tag}>'

    def table_head_parse(self, children=None):
        return Element(type='table_head', children=children)

    def table_head_render(self, el: Element):
        c = self.render_children(el)
        return f'<thead>\n<tr>{c}</tr>\n</thead>\n'

    def table_parse(self, children=None):
        return Element(type='table', children=children)

    def table_render(self, el: Element):
        c = self.render_children(el)
        return f'<table class="markdown-table">{c}</table>\n'

    def table_row_parse(self, children=None):
        return Element(type='table_row', children=children)

    def table_row_render(self, el: Element):
        c = self.render_children(el)
        return f'<tr>{c}</tr>\n'

    def text_parse(self, text):
        return Element(type='text', text=text)

    def text_render(self, el: Element):
        return escape(el.text)

    def thematic_break_parse(self):
        return Element(type='thematic_break')

    def thematic_break_render(self, el: Element):
        return '<hr/>\n'


def escape(s, quote=True):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    if quote:
        s = s.replace('"', "&quot;")
    return s


def attributes(attrs):
    s = ''
    if attrs:
        for k, v in attrs.items():
            s += f' {k}="{v}"'
    return s


##


_ATTRIBUTE_RE = r'''(?x)
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
'''


def parse_attributes(text):
    text = text.strip() + ' '
    res = {}

    while text:
        m = re.match(_ATTRIBUTE_RE, text)
        if not m:
            return {}

        text = text[m.end():].lstrip()

        g = m.groupdict()
        if g['id']:
            res['id'] = g['id']
        elif g['class']:
            res['class'] = (res.get('class', '') + ' ' + g['class']).strip()
        else:
            res[g['key']] = g['simple'] or g['quoted'].strip()

    return res
