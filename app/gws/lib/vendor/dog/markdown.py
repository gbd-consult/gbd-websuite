from typing import List

import mistune
from mistune import Markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters.html import HtmlFormatter

from . import util


class Element(util.Data):
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
    title: str
    type: str

    def __repr__(self):
        return repr(vars(self))


def parser() -> Markdown:
    return mistune.create_markdown(
        renderer=AstRenderer(),
        plugins=['table', 'url']
    )


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

    def register(self, name, method):
        pass
        # setattr(self, name, method)

    def text(self, text):
        return Element(type='text', text=text)

    def link(self, link, children=None, title=None):
        if isinstance(children, str):
            children = [Element(type='text', text=children)]
        return Element(
            type='link',
            link=link,
            children=children,
            title=title,
        )

    def image(self, src, alt="", title=None):
        return Element(type='image', src=src, alt=alt, title=title)

    def codespan(self, text):
        return Element(type='codespan', text=text)

    def linebreak(self):
        return Element(type='linebreak')

    def inline_html(self, html):
        return Element(type='inline_html', text=html)

    def heading(self, children, level):
        return Element(type='heading', children=children, level=level)

    def newline(self):
        return Element(type='newline')

    def thematic_break(self):
        return Element(type='thematic_break')

    def block_code(self, children, info=None):
        if info:
            try:
                lexer = get_lexer_by_name(info, stripall=True)
                formatter = HtmlFormatter(noclasses=True)
                res = highlight(children, lexer, formatter)
                return Element(
                    type='block_html',
                    text=res,
                    info=info
                )
            except:
                # no lexer found
                pass

        return Element(
            type='block_code',
            text=children,
            info=info
        )
        #return '<pre><code>' + mistune.escape(children) + '</code></pre>'

    def block_html(self, children):
        return Element(type='block_html', text=children)

    def list(self, children, ordered, level, start=None):
        token = {
            'type': 'list',
            'children': children,
            'ordered': ordered,
            'level': level,
        }
        if start is not None:
            token['start'] = start
        return Element(**token)

    def list_item(self, children, level):
        return Element(type='list_item', children=children, level=level)

    def table_cell(self, children, align=None, is_head=False):
        return Element(
            type='table_cell',
            children=children,
            align=align,
            is_head=is_head,
        )

    def _create_default_method(self, name):
        def __ast(children=None):
            return Element(type=name, children=children)

        return __ast

    def _get_method(self, name):
        try:
            return getattr(self, name)
        except AttributeError:
            return self._create_default_method(name)

    def finalize(self, data):
        return list(data)


##

class Renderer:

    def render_content(self, el: Element):
        if el.children:
            return ''.join(self.render_element(c) for c in el.children)
        return ''

    def render_element(self, el: Element):
        fn = getattr(self, 'tag_' + el.type)
        return fn(el)

    def render_a(self, href, title, content):
        s = f' href="{href}"'
        if title:
            s += f'title="{escape(title)}"'
        return f'<a{s}>{content or href}</a>'

    ##

    def tag_text(self, el: Element):
        return escape(el.text)

    def tag_link(self, el: Element):
        c = self.render_content(el)
        link = el.link
        return self.render_a(link, el.title, c)

    def tag_image(self, el: Element):
        s = ''
        if el.src:
            s += f' src="{el.src}"'
        if el.alt:
            s += f' alt="{escape(el.alt)}"'
        if el.title:
            s += f' title="{escape(el.title)}"'
        return f'<img{s} />'

    def tag_emphasis(self, el: Element):
        c = self.render_content(el)
        return f'<em>{c}</em>'

    def tag_strong(self, el: Element):
        c = self.render_content(el)
        return f'<strong>{c}</strong>'

    def tag_codespan(self, el: Element):
        return '<code>' + escape(el.text) + '</code>'

    def tag_linebreak(self, el: Element):
        return '<br />\n'

    def tag_inline_html(self, el: Element):
        return el.text

    def tag_paragraph(self, el: Element):
        c = self.render_content(el)
        return f'<p>{c}</p>\n'

    def tag_heading(self, el: Element):
        c = self.render_content(el)
        tag = 'h' + str(el.level)
        s = ''
        if el.id:
            s += f' id="{el.id}"'
        return f'<{tag}{s}>{c}</{tag}>\n'

    def tag_newline(self, el: Element):
        return ''

    def tag_thematic_break(self, el: Element):
        return '<hr/>\n'

    def tag_block_text(self, el: Element):
        return self.render_content(el)

    def tag_block_code(self, el: Element):
        # @TODO syntax highlighting
        c = escape(el.text.strip())
        return f'<pre><code>{c}</code></pre>'

    def tag_block_quote(self, el: Element):
        c = self.render_content(el)
        return f'<blockquote>\n{c}</blockquote>\n'

    def tag_block_html(self, el: Element):
        return el.text

    def tag_block_error(self, el: Element):
        c = self.render_content(el)
        return f'<div class="error">{c}</div>\n'

    def tag_list(self, el: Element):
        c = self.render_content(el)
        tag = 'ol' if el.ordered else 'ul'
        s = ''
        if el.start:
            s = f' start="{el.start}"'
        return f'<{tag}{s}>\n{c}</{tag}>\n'

    def tag_list_item(self, el: Element):
        c = self.render_content(el)
        return f'<li>{c}</li>\n'

    def tag_table(self, el: Element):
        c = self.render_content(el)
        return f'<table>{c}</table>\n'

    def tag_table_head(self, el: Element):
        c = self.render_content(el)
        return f'<thead>\n<tr>{c}</tr>\n</thead>\n'

    def tag_table_body(self, el: Element):
        c = self.render_content(el)
        return f'<tbody>\n{c}</tbody>\n'

    def tag_table_row(self, el: Element):
        c = self.render_content(el)
        return f'<tr>{c}</tr>\n'

    def tag_table_cell(self, el: Element):
        c = self.render_content(el)
        tag = 'th' if el.is_head else 'td'
        s = ''
        if el.align:
            s = f' style="text-align:{el.align}"'
        return f'<{tag}{s}>{c}</{tag}>'


def escape(s, quote=True):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    if quote:
        s = s.replace('"', "&quot;")
    return s
