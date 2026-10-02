import re
from typing import List

import mistune

import pygments
import pygments.util
import pygments.lexers
import pygments.formatters.html

from . import util as u
from .types import MarkdownElement, CssClasses

Markdown = mistune.Markdown


def parser() -> Markdown:
    md = mistune.create_markdown(renderer=AstRenderer(), plugins=['table', 'url', inline_decoration_plugin, link_attributes_plugin])
    return md


# plugin API reference: https://mistune.lepture.com/en/v2.0.5/advanced.html#create-plugins

# plugin: inline decorations
# {someclass some text} => <span class="md-decoration-someclass">some text</span>


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


def strip_text_content(el: MarkdownElement):
    while el.children:
        if not el.children[-1].text:
            return
        el.children[-1].text = el.children[-1].text.rstrip()
        if len(el.children[-1].text) > 0:
            return
        el.children.pop()


def longest_backtick_run(text: str) -> int:
    return max((len(s) for s in re.findall(r'`+', text)), default=0)


def text_from_element(el: MarkdownElement) -> str:
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

    def finalize(self, elements: List[MarkdownElement]):
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
        lang = ''
        atts = {}
        if info:
            # 'javascript' or 'javascript title=...' or 'title=...'
            m = re.match(r'^(\w+(?=(\s|$)))?(.*)$', info.strip())
            if m:
                lang = m.group(1) or ''
                atts = parse_attributes(m.group(3))
        return MarkdownElement(
            type='block_code',
            text=u.strip_blank_lines(text),
            lang=lang,
            title=atts.get('title'),
            attributes=atts,
        )

    def p_block_error(self, children=None):
        return MarkdownElement(type='block_error', children=children)

    def p_block_html(self, html):
        return MarkdownElement(type='block_html', html=html)

    def p_block_quote(self, children=None):
        return MarkdownElement(type='block_quote', children=children)

    def p_block_text(self, children=None):
        return MarkdownElement(type='block_text', children=children)

    def p_codespan(self, text):
        return MarkdownElement(type='codespan', text=text)

    def p_emphasis(self, children):
        return MarkdownElement(type='emphasis', children=children)

    def p_heading(self, children, level):
        return MarkdownElement(type='heading', children=children, level=level)

    def p_image(self, src, alt='', title=None):
        return MarkdownElement(type='image', src=src, alt=alt, title=title)

    def p_inline_decoration(self, classname, text):
        return MarkdownElement(type='inline_decoration', classname=classname, text=text)

    def p_inline_html(self, html):
        return MarkdownElement(type='inline_html', html=html)

    def p_linebreak(self):
        return MarkdownElement(type='linebreak')

    def p_link(self, target, children=None, title=None):
        if isinstance(children, str):
            children = [MarkdownElement(type='text', text=children)]
        return MarkdownElement(type='link', target=target, children=children, title=title)

    def p_link_attributes(self, text, attributes):
        return MarkdownElement(type='link_attributes', text=text, attributes=attributes)

    def p_list_item(self, children, level):
        return MarkdownElement(type='list_item', children=children, level=level)

    def p_list(self, children, ordered, level, start=None):
        return MarkdownElement(type='list', children=children, ordered=ordered, level=level, start=start)

    def p_newline(self):
        return MarkdownElement(type='newline')

    def p_paragraph(self, children=None):
        return MarkdownElement(type='paragraph', children=children)

    def p_strong(self, children=None):
        return MarkdownElement(type='strong', children=children)

    def p_table_body(self, children=None):
        return MarkdownElement(type='table_body', children=children)

    def p_table_cell(self, children, align=None, is_head=False):
        return MarkdownElement(type='table_cell', children=children, align=align, isTableHead=is_head)

    def p_table_head(self, children=None):
        return MarkdownElement(type='table_head', children=children)

    def p_table(self, children=None):
        return MarkdownElement(type='table', children=children)

    def p_table_row(self, children=None):
        return MarkdownElement(type='table_row', children=children)

    def p_text(self, text):
        return MarkdownElement(type='text', text=text)

    def p_thematic_break(self):
        return MarkdownElement(type='thematic_break')


class _Renderer:
    def render_children(self, el: MarkdownElement):
        if el.children:
            return ''.join(self.render_element(c) for c in el.children)
        return ''

    def render_element(self, el: MarkdownElement):
        fn = getattr(self, f'r_{el.type}')
        return fn(el)


class MarkdownRenderer(_Renderer):
    list_ordered = False
    list_index = 1

    def render_link(self, href, title, content, el):
        title = f' "{title}"' if title else ''
        return f'[{content}]({href}{title})'

    def r_block_code(self, el: MarkdownElement):
        fence = '`' * max(3, longest_backtick_run(el.text) + 1)
        code = f'{fence}{el.lang or ""}\n{el.text}\n{fence}\n'
        if el.title:
            code = f'**{el.title}**\n\n' + code
        return code

    def r_block_error(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'> **ERROR:** {c}\n\n'

    def r_block_html(self, el: MarkdownElement):
        return el.html + '\n\n'

    def r_block_quote(self, el: MarkdownElement):
        c = self.render_children(el)
        lines = c.split('\n')
        return ''.join(f'> {line}\n' for line in lines) + '\n'

    def r_block_text(self, el: MarkdownElement):
        return self.render_children(el)

    def r_codespan(self, el: MarkdownElement):
        return f'`{el.text}`'

    def r_emphasis(self, el: MarkdownElement):
        return f'*{self.render_children(el)}*'

    def r_heading(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'{"#" * el.level} {c}\n\n'

    def r_image(self, el: MarkdownElement):
        title = f' "{el.title}"' if el.title else ''
        return f'![{el.alt or ""}]({el.src}{title})'

    def r_inline_decoration(self, el: MarkdownElement):
        return f'{{{el.classname} {el.text}}}'

    def r_inline_html(self, el: MarkdownElement):
        return el.html

    def r_linebreak(self, el: MarkdownElement):
        return '\n'

    def r_link(self, el: MarkdownElement):
        c = self.render_children(el)
        return self.render_link(el.target, el.title, c or el.target, el)

    def r_list_item(self, el: MarkdownElement):
        c = self.render_children(el)
        indent = '  ' * (el.level - 1)
        if self.list_ordered:
            marker = f'{self.list_index}. '
            self.list_index += 1
        else:
            marker = '- '
        return f'{indent}{marker}{c}\n'

    def r_list(self, el: MarkdownElement):
        prev = self.list_ordered, self.list_index
        self.list_ordered = bool(el.ordered)
        self.list_index = int(el.start or 1)
        c = self.render_children(el)
        self.list_ordered, self.list_index = prev
        return c + '\n'

    def r_newline(self, el: MarkdownElement):
        return '\n'

    def r_paragraph(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'{c}\n\n'

    def r_strong(self, el: MarkdownElement):
        return f'**{self.render_children(el)}**'

    def r_table(self, el: MarkdownElement):
        return self.render_children(el) + '\n'

    def r_table_head(self, el: MarkdownElement):
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

    def r_table_body(self, el: MarkdownElement):
        return self.render_children(el)

    def r_table_row(self, el: MarkdownElement):
        cells = [child for child in el.children if child.type == 'table_cell']
        return '| ' + ' | '.join(self.render_children(cell) for cell in cells) + ' |\n'

    def r_table_cell(self, el: MarkdownElement):
        return self.render_children(el)

    def r_text(self, el: MarkdownElement):
        return el.text

    def r_thematic_break(self, el: MarkdownElement):
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

    def r_block_code(self, el: MarkdownElement):
        lang = el.lang or 'text'
        try:
            lexer = pygments.lexers.get_lexer_by_name(lang, stripall=True)
        except pygments.util.ClassNotFound:
            u.log.warning(f'pygments lexer {lang!r} not found')
            lexer = pygments.lexers.get_lexer_by_name('text', stripall=True)

        kwargs = {}
        if 'numbers' in el.attributes:
            kwargs['linenos'] = 'table'
            kwargs['linenostart'] = el.attributes['numbers']

        formatter = pygments.formatters.html.HtmlFormatter(
            noclasses=True,
            nobackground=True,
            cssclass=CssClasses.CODE_BLOCK,
            **kwargs,
        )
        html = pygments.highlight(el.text, lexer, formatter)

        if el.title:
            html = f'<div class="{CssClasses.CODE_BLOCK_TITLE}">{escape(el.title)}</div>' + html

        return html

    def r_block_error(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'<div class="{CssClasses.ERROR}">{c}</div>\n'

    def r_block_html(self, el: MarkdownElement):
        return el.html

    def r_block_quote(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'<blockquote>\n{c}</blockquote>\n'

    def r_block_text(self, el: MarkdownElement):
        return self.render_children(el)

    def r_codespan(self, el: MarkdownElement):
        c = escape(el.text)
        return f'<code{attributes(el.attributes)}>{c}</code>'

    def r_emphasis(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'<em>{c}</em>'

    def r_heading(self, el: MarkdownElement):
        c = self.render_children(el)
        tag = 'h' + str(el.level)
        s = ''
        if el.htmlId:
            s += f' id="{el.htmlId}"'
        return f'<{tag}{s}>{c}</{tag}>\n'

    def r_image(self, el: MarkdownElement):
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

    def r_inline_decoration(self, el: MarkdownElement):
        c = escape(el.text)
        return f'<span class="{CssClasses.DECORATION}{el.classname}">{c}</span>'

    def r_inline_html(self, el: MarkdownElement):
        return el.html

    def r_linebreak(self, el: MarkdownElement):
        return '<br/>\n'

    def r_link(self, el: MarkdownElement):
        c = self.render_children(el)
        return self.render_link(el.target, el.title, c, el)

    def r_list_item(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'<li>{c}</li>\n'

    def r_list(self, el: MarkdownElement):
        c = self.render_children(el)
        tag = 'ol' if el.ordered else 'ul'
        a = {}
        if el.start:
            a['start'] = el.start
        return f'<{tag}{attributes(a)}>\n{c}\n</{tag}>\n'

    def r_newline(self, el: MarkdownElement):
        return ''

    def r_paragraph(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'<p>{c}</p>\n'

    def r_strong(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'<strong>{c}</strong>'

    def r_table_body(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'<tbody>\n{c}</tbody>\n'

    def r_table_cell(self, el: MarkdownElement):
        c = self.render_children(el)
        tag = 'th' if el.isTableHead else 'td'
        a = {}
        if el.align:
            a['style'] = f'text-align:{el.align}'
        return f'<{tag}{attributes(a)}>{c}</{tag}>'

    def r_table_head(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'<thead>\n<tr>{c}</tr>\n</thead>\n'

    def r_table(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'<table class="{CssClasses.TABLE}">{c}</table>\n'

    def r_table_row(self, el: MarkdownElement):
        c = self.render_children(el)
        return f'<tr>{c}</tr>\n'

    def r_text(self, el: MarkdownElement):
        return escape(el.text)

    def r_thematic_break(self, el: MarkdownElement):
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
