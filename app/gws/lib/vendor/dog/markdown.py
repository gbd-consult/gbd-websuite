from typing import List
import mistune

from . import types as t
from . import util


def parser():
    return mistune.create_markdown(
        renderer=AstRenderer(),
        plugins=['table', 'url']
    )


def strip_node(node: t.MarkdownNode):
    while node.children:
        if not node.children[-1].text:
            return
        node.children[-1].text = node.children[-1].text.rstrip()
        if len(node.children[-1].text) > 0:
            return
        node.children.pop()


def text_from_node(node: t.MarkdownNode):
    if node.text:
        return node.text.strip()
    if node.children:
        return ' '.join(text_from_node(c) for c in node.children).strip()
    return ''


# based on mistune/renderers.AstRenderer

class AstRenderer:
    NAME = 'ast'

    def register(self, name, method):
        pass
        # setattr(self, name, method)

    def text(self, text):
        return t.MarkdownNode(type='text', text=text)

    def link(self, link, children=None, title=None):
        if isinstance(children, str):
            children = [t.MarkdownNode(type='text', text=children)]
        return t.MarkdownNode(
            type='link',
            link=link,
            children=children,
            title=title,
        )

    def image(self, src, alt="", title=None):
        return t.MarkdownNode(type='image', src=src, alt=alt, title=title)

    def codespan(self, text):
        return t.MarkdownNode(type='codespan', text=text)

    def linebreak(self):
        return t.MarkdownNode(type='linebreak')

    def inline_html(self, html):
        return t.MarkdownNode(type='inline_html', text=html)

    def heading(self, children, level):
        return t.MarkdownNode(type='heading', children=children, level=level)

    def newline(self):
        return t.MarkdownNode(type='newline')

    def thematic_break(self):
        return t.MarkdownNode(type='thematic_break')

    def block_code(self, children, info=None):
        return t.MarkdownNode(
            type='block_code',
            text=children,
            info=info
        )

    def block_html(self, children):
        return t.MarkdownNode(type='block_html', text=children)

    def list(self, children, ordered, level, start=None):
        token = {
            'type': 'list',
            'children': children,
            'ordered': ordered,
            'level': level,
        }
        if start is not None:
            token['start'] = start
        return t.MarkdownNode(**token)

    def list_item(self, children, level):
        return t.MarkdownNode(type='list_item', children=children, level=level)

    def table_cell(self, children, align=None, is_head=False):
        return t.MarkdownNode(
            type='table_cell',
            children=children,
            align=align,
            is_head=is_head,
        )

    def _create_default_method(self, name):
        def __ast(children=None):
            return t.MarkdownNode(type=name, children=children)

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

    def render_content(self, node):
        if node.children:
            return ''.join(self.render_node(c) for c in node.children)
        return ''

    def render_node(self, node):
        fn = getattr(self, 'tag_' + node.type)
        return fn(node)

    def render_a(self, href, title, content):
        s = f' href="{href}"'
        if title:
            s += f'title="{escape(title)}"'
        return f'<a{s}>{content or href}</a>'

    ##

    def tag_text(self, node):
        return escape(node.text)

    def tag_link(self, node):
        c = self.render_content(node)
        link = node.link
        return self.render_a(link, node.title, c)

    def tag_image(self, node):
        s = ''
        if node.src:
            s += f' src="{node.src}"'
        if node.alt:
            s += f' alt="{escape(node.alt)}"'
        if node.title:
            s += f' title="{escape(node.title)}"'
        return f'<img{s} />'

    def tag_emphasis(self, node):
        c = self.render_content(node)
        return f'<em>{c}</em>'

    def tag_strong(self, node):
        c = self.render_content(node)
        return f'<strong>{c}</strong>'

    def tag_codespan(self, node):
        return '<code>' + escape(node.text) + '</code>'

    def tag_linebreak(self, node):
        return '<br />\n'

    def tag_inline_html(self, node):
        return self.render_content(node)

    def tag_paragraph(self, node):
        c = self.render_content(node)
        return f'<p>{c}</p>\n'

    def tag_heading(self, node):
        c = self.render_content(node)
        tag = 'h' + str(node.level)
        s = ''
        if node.id:
            s += f' id="{node.id}"'
        return f'<{tag}{s}>{c}</{tag}>\n'

    def tag_newline(self, node):
        return ''

    def tag_thematic_break(self, node):
        return '<hr/>\n'

    def tag_block_text(self, node):
        return self.render_content(node)

    def tag_block_code(self, node):
        # @TODO syntax higlighting
        c = escape(node.text.strip())
        return f'<pre><code>{c}</code></pre>'

    def tag_block_quote(self, node):
        c = self.render_content(node)
        return f'<blockquote>\n{c}</blockquote>\n'

    def tag_block_html(self, node):
        return node.text

    def tag_block_error(self, node):
        c = self.render_content(node)
        return f'<div class="error">{c}</div>\n'

    def tag_list(self, node):
        c = self.render_content(node)
        tag = 'ol' if node.ordered else 'ul'
        s = ''
        if node.start:
            s = f' start="{node.start}"'
        return f'<{tag}{s}>\n{c}</{tag}>\n'

    def tag_list_item(self, node):
        c = self.render_content(node)
        return f'<li>{c}</li>\n'

    def tag_table(self, node):
        c = self.render_content(node)
        return f'<table>{c}</table>\n'

    def tag_table_head(self, node):
        c = self.render_content(node)
        return f'<thead>\n<tr>{c}</tr>\n</thead>\n'

    def tag_table_body(self, node):
        c = self.render_content(node)
        return f'<tbody>\n{c}</tbody>\n'

    def tag_table_row(self, node):
        c = self.render_content(node)
        return f'<tr>{c}</tr>\n'

    def tag_table_cell(self, node):
        c = self.render_content(node)
        tag = 'th' if node.is_head else 'td'
        s = ''
        if node.align:
            s = f' style="text-align:{node.align}"'
        return f'<{tag}{s}>{c}</{tag}>'


def escape(s, quote=True):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    if quote:
        s = s.replace('"', "&quot;")
    return s
