from typing import List, Optional

import re
import os
import json
import fnmatch
import shutil
import mimetypes

from . import util, template, markdown


class ParseNode(util.Data):
    pass


class MarkdownNode(ParseNode):
    el: markdown.Element


class SectionNode(ParseNode):
    sid: str


class EmbedNode(ParseNode):
    items: List[str]
    sid: str


class TocNode(ParseNode):
    items: List[str]
    sids: List[str]
    depth: int


class RawHtmlNode(ParseNode):
    html: str


class Section(util.Data):
    sid: str
    level: int

    subSids: List[str]
    parentSid: str

    sourcePath: str

    headText: str
    headHtml: str
    headHtmlLink: str
    headNode: MarkdownNode
    headLevel: int

    nodes: List[ParseNode]

    htmlPath: str
    htmlUrl: str
    htmlBaseUrl: str
    htmlId: str

    walkColor: int


class Builder:
    options: util.Data

    def __init__(self, options):
        self.options = util.to_data(options)

        self.docPaths = []
        self.assetPaths = []

        self.markdownParser = markdown.parser()
        self.htmlGenerator = HTMLGenerator(self)

        self.sectionMap = {}
        self.assetMap = {}

        self.debug = options.debug
        util.log.set_level('DEBUG' if options.verbose else 'INFO')

        self.includeTemplate = ''
        if self.options.includeTemplate:
            self.includeTemplate = util.read_file(self.options.includeTemplate)

    def build_all(self, mode: str, write=False):
        util.log.debug(f'START build_all {mode}')

        self.sectionMap = {sec.sid: sec for sec in self.parse_all()}

        for sec in self.sectionMap.values():
            self.make_tree(sec)

        for sec in self.sectionMap.values():
            self.add_url_and_path(sec)

        if write:
            os.makedirs(self.options.outputDir, exist_ok=True)

        if write:
            self.dump({'sec': list(self.sectionMap.values())})

        if mode == 'html':
            self.generate_html(write)

        util.log.debug(f'END build_all {mode}')

    def parse_all(self):
        self.collect_sources()

        return [
            sec
            for path in self.docPaths
            for sec in self.parse_file(path)
        ]

    def parse_file(self, path):
        return FileParser(self, path).parse()

    def collect_sources(self):
        for dirname in self.options.rootDirs:
            self.collect_sources_from_dir(dirname)

    def collect_sources_from_dir(self, dirname):
        de: os.DirEntry
        ex = self.options.excludeRegex

        for de in os.scandir(dirname):
            if de.name.startswith('.'):
                pass
            elif ex and re.search(ex, de.path):
                util.log.debug(f'exclude: {de.path!r}')
            elif de.is_dir():
                self.collect_sources_from_dir(de.path)
            elif de.is_file() and any(fnmatch.fnmatch(de.name, p) for p in self.options.docPatterns):
                self.docPaths.append(de.path)
            elif de.is_file() and any(fnmatch.fnmatch(de.name, p) for p in self.options.assetPatterns):
                self.assetPaths.append(de.path)

    ##

    def get_section(self, sid) -> Optional[Section]:
        if sid not in self.sectionMap:
            util.log.error(f'section {sid!r} not found')
            return
        return self.sectionMap.get(sid)

    def section_from_url(self, url) -> Optional[Section]:
        for sec in self.sectionMap.values():
            if sec.htmlBaseUrl == url:
                return sec

    def sections_from_wildcard_sid(self, sid, parent_sec) -> List[Section]:
        abs_sid = self.make_sid(sid, parent_sec.sid, '', '')

        if not abs_sid:
            util.log.error(f'invalid section id {sid!r} in {parent_sec.sourcePath!r}')
            return []

        if '*' not in abs_sid:
            sub = self.get_section(abs_sid)
            if sub:
                return [sub]
            return []

        rx = abs_sid.replace('*', '[^/]+') + '$'
        subs = [
            sec
            for sec in self.sectionMap.values()
            if re.match(rx, sec.sid)
        ]
        return sorted(subs, key=lambda sec: sec.headText)

    ##

    def generate_html(self, write):
        self.assetMap = {}

        for path in self.options.extraAssets:
            self.add_asset(path)

        self.htmlGenerator = HTMLGenerator(self)
        self.htmlGenerator.render_section_heads()
        self.htmlGenerator.render_sections()
        self.htmlGenerator.flush()

        if write:
            self.htmlGenerator.write()
            self.write_assets()

    ##

    def content_for_url(self, url):
        if url.endswith('.html'):
            sec = self.section_from_url(url)
            if sec:
                return 'text/html', self.htmlGenerator.content[sec.htmlPath]
            return

        m = re.search(self.options.staticDir + '/(.+)$', url)
        if not m:
            return
        fn = m.group(1)
        for path, fname in self.assetMap.items():
            if fname == fn:
                mt = mimetypes.guess_type(path)
                return mt[0] if mt else 'text/plain', util.read_file_b(path)

    def add_asset(self, path):
        if path not in self.assetMap:
            self.assetMap[path] = self.unique_asset_filename(path)
        return self.options.webRoot + '/' + self.options.staticDir + '/' + self.assetMap[path]

    def unique_asset_filename(self, path):
        fnames = set(self.assetMap.values())
        fname = os.path.basename(path)
        if fname not in fnames:
            return fname
        n = 1
        while True:
            base, ext = fname.split('.')
            fname2 = f'{base}-{n}.{ext}'
            if fname2 not in fnames:
                return fname2
            n += 1

    def write_assets(self):
        dirname = self.options.outputDir + '/' + self.options.staticDir
        os.makedirs(dirname, exist_ok=True)
        for src, fname in self.assetMap.items():
            dst = dirname + '/' + fname
            util.log.debug(f'copy {src!r} => {dst!r}')
            shutil.copy(src, dst)

    ##

    def make_tree(self, sec: Section, parent_sec=None):
        if parent_sec:
            if sec.parentSid:
                util.log.error(f'attempt to relink section {sec.sid!r} for {sec.parentSid!r} to {parent_sec.sid!r}')
            sec.parentSid = parent_sec.sid

        if sec.walkColor == 2:
            return

        if sec.walkColor == 1:
            util.log.error(f'circular dependency in {sec.sid!r}')
            return

        cur_nodes = sec.nodes

        sec.subSids = []
        sec.nodes = []
        sec.walkColor = 1

        for node in cur_nodes:

            if isinstance(node, SectionNode):
                sub = self.get_section(node.sid)
                if sub:
                    self.make_tree(sub, sec)
                    sec.subSids.append(sub.sid)
                    sec.nodes.append(node)
                continue

            if isinstance(node, EmbedNode):
                secs = self.sections_from_wildcard_sid(node.sid, sec)
                for sub in secs:
                    self.make_tree(sub, sec)
                    sec.subSids.append(sub.sid)
                    sec.nodes.append(SectionNode(sid=sub.sid))
                continue

            if isinstance(node, TocNode):
                sids = []
                for sid in node.items:
                    secs = self.sections_from_wildcard_sid(sid, sec)
                    sids.extend(s.sid for s in secs)
                sec.nodes.append(TocNode(depth=node.depth, sids=sids))
                continue

            sec.nodes.append(node)

        sec.walkColor = 2

    def add_url_and_path(self, sec: Section):
        sl = self.options.htmlSplitLevel or 0
        parts = sec.sid.split('/')[1:]

        if sec.level == 0 or sl == 0:
            path = 'index.html'
        else:
            dirname = '/'.join(parts[:sl])
            path = dirname + '/index.html'

        sec.htmlId = '-'.join(parts[sl:])
        sec.htmlPath = self.options.outputDir + '/' + path
        sec.htmlBaseUrl = self.options.webRoot + '/' + path

        sec.htmlUrl = sec.htmlBaseUrl
        if sec.htmlId:
            sec.htmlUrl += '#' + sec.htmlId

        sec.headLevel = max(1, sec.level - sl + 1)

    def make_sid(self, explicit_sid, parent_sid, prev_sid=None, text=None):

        explicit_sid = explicit_sid or ''
        text_sid = util.to_uid(text) if text else ''

        if explicit_sid == '/':
            return '/'

        sid = explicit_sid or text_sid
        if sid.endswith('/'):
            sid += text_sid
        if not sid or sid.endswith('/'):
            return ''

        if sid.startswith('/'):
            return util.normpath(sid)

        if parent_sid:
            return util.normpath(parent_sid + '/' + sid)

        if prev_sid:
            ps, _, _ = prev_sid.rpartition('/')
            return util.normpath(ps + '/' + sid)

        return ''

    ##

    def dump(self, obj):
        def dflt(x):
            d = dict(vars(x))
            d['$'] = x.__class__.__name__
            return d

        r = json.dumps(obj, indent=4, default=dflt)
        with open(self.options.outputDir + '/parser.json', 'w') as fp:
            fp.write(r)


class FileParser:
    def __init__(self, b: Builder, path):
        self.b = b
        self.path = path
        self.dummyRoot = Section(sid='', nodes=[], level=-1, headNode=MarkdownNode(el=markdown.Element(level=-1)))

    def parse(self) -> List[Section]:
        util.log.debug(f'parse {self.path!r}')

        src = self.pre_parse()
        if not src:
            return []

        sections = []
        stack = [self.dummyRoot]

        el: markdown.Element
        for el in self.b.markdownParser(src):

            if el.type == 'heading':
                prev_sec = None
                while stack[-1].headNode.el.level > el.level:
                    stack.pop()
                if stack[-1].headNode.el.level == el.level:
                    prev_sec = stack.pop()

                sec = self.parse_heading(el, stack[-1], prev_sec)
                if sec:
                    stack.append(sec)
                    sections.append(sec)

                continue

            if el.type == 'block_code' and el.text.startswith(template.COMMAND):
                args = json.loads(el.text[len(template.COMMAND):])
                cls = globals()[args['command']]
                stack[-1].nodes.append(cls(**args))
                continue

            stack[-1].nodes.append(MarkdownNode(el=el))

        return sections

    def pre_parse(self):
        text = self.b.includeTemplate + util.read_file(self.path)
        return template.render(text, self.path, {
            'options': self.b.options,
            'builder': self.b,
        })

    def parse_heading(self, el: markdown.Element, parent_sec, prev_sec):
        explicit_sid = self.extract_explicit_sid(el)
        text = markdown.text_from_element(el)

        sid = self.b.make_sid(
            explicit_sid,
            parent_sec.sid,
            prev_sec.sid if prev_sec else None,
            text
        )

        if not sid:
            util.log.error(f'invalid section id for {text!r} in {self.path!r}')
            return

        if not text:
            parent_sec.nodes.append(EmbedNode(sid=sid))
            return

        parent_sec.nodes.append(SectionNode(sid=sid))
        el.sid = sid
        head_node = MarkdownNode(el=el)

        return Section(
            sid=sid,
            level=0 if sid == '/' else sid.count('/'),
            sourcePath=self.path,
            headText=text,
            headNode=head_node,
            nodes=[head_node],
        )

    def extract_explicit_sid(self, el: markdown.Element) -> str:
        ch = el.children

        if not ch or ch[-1].type != 'text':
            return ''

        m = re.match(r'^(.*?):(\S+)$', ch[-1].text)
        if not m:
            return ''

        ch[-1].text = m.group(1)
        markdown.strip_text_content(el)

        return m.group(2)


class HTMLGenerator:
    def __init__(self, b: Builder):
        self.b = b
        self.buffers = {}
        self.content = {}

    def render_section_heads(self):
        for sec in self.b.sectionMap.values():
            mr = MarkdownRenderer(self.b, sec)
            sec.headHtml = mr.render_content(sec.headNode.el)
            sec.headHtmlLink = f'<a href="{sec.htmlUrl}">{sec.headHtml}</a>'

    def render_sections(self):
        for sec in self.b.sectionMap.values():
            if not sec.parentSid:
                self.render_section(sec.sid)

    def render_section(self, sid):
        sec = self.b.get_section(sid)
        if not sec:
            return

        util.log.debug(f'render {sid!r}')

        mr = MarkdownRenderer(self.b, sec)

        self.add(sec, f'<section id="{sec.htmlId}" data-sid="{sec.sid}">\n')

        for node in sec.nodes:
            if isinstance(node, MarkdownNode):
                html = mr.render_element(node.el)
                self.add(sec, html)
                continue
            if isinstance(node, SectionNode):
                self.render_section(node.sid)
                continue
            if isinstance(node, TocNode):
                entries = ''.join(self.render_toc_entry(sid, node.depth) for sid in node.sids)
                html = f'<div class="localtoc"><ul>{entries}</ul></div>'
                self.add(sec, html)
                continue
            if isinstance(node, RawHtmlNode):
                self.add(sec, node.html)
                continue

        self.add(sec, f'</section>\n')

    def render_toc_entry(self, sid, depth: int):
        sec = self.b.get_section(sid)
        if not sec:
            return

        s = ''
        if depth > 1:
            sub = [self.render_toc_entry(s, depth - 1) for s in sec.subSids]
            if sub:
                s = '<ul>' + ''.join(sub) + '</ul>'

        return f'<li data-sid="{sid}">{sec.headHtmlLink}{s}</li>'

    def render_main_toc(self):
        root = self.b.get_section('/')
        if not root:
            return
        return '\n'.join(
            self.render_toc_entry(sid, 999)
            for sid in root.subSids
        )

    def add(self, sec: Section, html):
        if sec.htmlPath not in self.buffers:
            self.buffers[sec.htmlPath] = util.Data(sids=[], html=[])
        self.buffers[sec.htmlPath].sids.append(sec.sid)
        self.buffers[sec.htmlPath].html.append(html)

    def flush(self):
        tpl = template.compile(self.b.options.pageTemplate)
        maintoc = self.render_main_toc()

        self.content = {}

        for path, buf in self.buffers.items():
            self.content[path] = template.call(tpl, {
                'path': path,
                'title': self.b.options.title,
                'subTitle': self.b.options.subTitle,
                'mainToc': maintoc,
                'main': ''.join(buf.html),
                'breadcrumbs': self.get_breadcrumbs(buf.sids[0]),
                'builder': self,
                'options': self.b.options,
            })

    def write(self):
        for path, html in self.content.items():
            util.log.debug(f'write {path!r}')
            dirs = os.path.dirname(path)
            os.makedirs(dirs, exist_ok=True)
            util.write_file(path, html)

    def get_breadcrumbs(self, sid):
        sec = self.b.get_section(sid)
        if not sec:
            return []

        bs = []

        while sec:
            bs.insert(0, (sec.htmlUrl, sec.headHtml))
            if not sec.parentSid:
                break
            sec = self.b.get_section(sec.parentSid)

        return bs


class MarkdownRenderer(markdown.Renderer):

    def __init__(self, b: Builder, sec: Section):
        self.b = b
        self.sec = sec

    def tag_link(self, el: markdown.Element):
        c = self.render_content(el)
        link = el.link
        if link.startswith(('http:', 'https:')):
            return self.render_a(link, el.title, c)
        if link.startswith('//'):
            return self.render_a(link[1:], el.title, c)

        sid = self.b.make_sid(link, self.sec.sid)
        target = self.b.get_section(sid)
        if not target:
            return self.render_a(link, el.title, c)

        return self.render_a(
            target.htmlUrl,
            el.title or target.headText,
            c or target.headHtml)

    def tag_image(self, el: markdown.Element):
        if not el.src:
            return ''
        if el.src.startswith(('http:', 'https:')):
            return super().tag_image(el)
        paths = [path for path in self.b.assetPaths if path.endswith(el.src)]
        if not paths:
            util.log.error(f'image {el.src!r} not found')
            return super().tag_image(el)
        el.src = self.b.add_asset(paths[0])
        return super().tag_image(el)

    def tag_heading(self, el: markdown.Element):
        # NB heading sid must set in `parse_heading` 
        sec = self.b.get_section(el.sid)
        if not sec:
            return

        c = self.render_content(el)
        tag = 'h' + str(sec.headLevel)
        s = ''
        if self.b.debug:
            s = f' title="{markdown.escape(sec.sourcePath)}"'
        a = f'<a class="header-link" href="{sec.htmlUrl}">&para;</a>'
        return f'<{tag}{s}>{c}{a}</{tag}>\n'
