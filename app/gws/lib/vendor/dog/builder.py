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
    status: str
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


class Builder:
    options: util.Data
    markdownParser: markdown.Markdown
    htmlGenerator: 'HTMLGenerator'
    docPaths: set[str]
    assetPaths: set[str]
    sectionMap: dict[str, Section]
    sectionNotFound: set[str]
    assetMap: dict[str, str]

    def __init__(self, options):
        self.options = util.to_data(options)

        util.log.set_level('DEBUG' if options.verbose else 'INFO')

        self.includeTemplate = ''
        if self.options.includeTemplate:
            self.includeTemplate = util.read_file(self.options.includeTemplate)

        self.cache = {}

    def collect_and_parse(self):
        self.markdownParser = markdown.parser()

        self.docPaths = set()
        self.assetPaths = set()
        self.sectionMap = {}
        self.sectionNotFound = set()
        self.assetMap = {}

        self.collect_sources()
        self.parse_all()

    def build_html(self, write=False):
        self.collect_and_parse()
        self.generate_html(write=write)

    def build_pdf(self):
        pdf_dir = '/tmp/dogpdf'
        shutil.rmtree(pdf_dir, ignore_errors=True)

        old_opts = util.to_data(self.options)

        self.options.htmlSplitLevel = 0
        self.options.outputDir = pdf_dir
        self.options.webRoot = '.'

        self.collect_and_parse()
        self.generate_html(write=True)

        self.options = old_opts

        self.generate_pdf(
            pdf_dir + '/index.html',
            self.options.outputDir + '/index.pdf')

        shutil.rmtree(pdf_dir, ignore_errors=True)

    def dump(self):
        def _default(x):
            d = dict(vars(x))
            d['$'] = x.__class__.__name__
            return d

        self.collect_and_parse()
        return json.dumps(
            self.sectionMap, indent=4, sort_keys=True, ensure_ascii=False, default=_default)

    ##

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
                self.docPaths.add(de.path)
            elif de.is_file() and any(fnmatch.fnmatch(de.name, p) for p in self.options.assetPatterns):
                self.assetPaths.add(de.path)

    ##

    def get_section(self, sid: str) -> Optional[Section]:
        if sid in self.sectionNotFound:
            return
        if sid not in self.sectionMap:
            util.log.error(f'section not found: {sid!r}')
            self.sectionNotFound.add(sid)
            return
        return self.sectionMap.get(sid)

    def section_from_url(self, url) -> Optional[Section]:
        for sec in self.sectionMap.values():
            if sec.htmlBaseUrl == url:
                return sec

    def section_from_element(self, el: markdown.Element) -> Optional[Section]:
        for sec in self.sectionMap.values():
            if sec.headNode.el == el:
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

            util.write_file(
                str(os.path.join(self.options.outputDir, self.options.staticDir, self.GLOBAL_TOC_SCRIPT)),
                self.generate_global_toc())

            util.write_file(
                str(os.path.join(self.options.outputDir, self.options.staticDir, self.SEARCH_INDEX_SCRIPT)),
                self.generate_search_index())

    def generate_pdf(self, source: str, target: str):
        cmd = [
            'wkhtmltopdf',
            '--outline',
            '--enable-local-file-access',
            '--print-media-type',
            '--disable-javascript',
        ]

        if self.options.pdfOptions:
            for k, v in self.options.pdfOptions.items():
                cmd.append(f'--{k}')
                if v is not True:
                    cmd.append(str(v))

        cmd.append(source)
        cmd.append(target)

        util.run(cmd, pipe=True)

    ##

    GLOBAL_TOC_SCRIPT = '_global_toc.js'
    SEARCH_INDEX_SCRIPT = '_search_index.js'

    def generate_global_toc(self):
        js = {
            sec.sid: {
                'h': sec.headText,
                'u': sec.htmlUrl,
                'p': '',
                's': sec.subSids
            }
            for sec in self.sectionMap.values()
        }
        for sec in self.sectionMap.values():
            for sub in sec.subSids:
                node = js.get(sub)
                if node:
                    node['p'] = sec.sid

        return 'GLOBAL_TOC = ' + json.dumps(js, ensure_ascii=False, indent=4) + '\n'

    def generate_search_index(self):
        words_map = {}

        for sec in self.sectionMap.values():
            words_map[sec.sid] = []
            for node in sec.nodes:
                if isinstance(node, MarkdownNode):
                    self.extract_text(node.el, words_map[sec.sid])

        for sid, words in words_map.items():
            ws = ' '.join(words)
            ws = ws.replace("'", '')
            ws = re.sub(r'\W+', ' ', ws).lower().strip()
            words_map[sid] = ws.split()

        all_words = sorted(set(w for ws in words_map.values() for w in ws))
        word_index = {w: n for n, w in enumerate(all_words, 1)}

        sections = []
        for sid, words in words_map.items():
            sec = self.sectionMap[sid]
            head = sec.headHtml
            if sec.parentSid:
                parent = self.sectionMap[sec.parentSid]
                head += ' (' + parent.headHtml + ')'
            sections.append({
                'h': head,
                'u': sec.htmlUrl,
                'w': '.' + '.'.join(util.base36(word_index[w]) for w in words) + '.'
            })

        js = {
            'words': '.' + '.'.join(all_words),
            'sections': sorted(sections, key=lambda s: s['h']),
        }

        return 'SEARCH_INDEX = ' + json.dumps(js, ensure_ascii=False, indent=4) + '\n'

    def extract_text(self, el: markdown.Element, out: list):
        if el.text:
            out.append(el.text)
            return
        if el.children:
            for c in el.children:
                self.extract_text(c, out)
            out.append('.')

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
        if fn.endswith(self.GLOBAL_TOC_SCRIPT):
            return 'application/javascript', self.generate_global_toc()

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
        for src, fname in self.assetMap.items():
            dst = str(os.path.join(self.options.outputDir, self.options.staticDir, fname))
            util.log.debug(f'copy {src!r} => {dst!r}')
            util.write_file_b(dst, util.read_file_b(src))

    ##

    def parse_all(self):
        self.sectionMap = {}

        for path in self.docPaths:
            for sec in self.parse_file(path):
                prev = self.sectionMap.get(sec.sid)
                if prev:
                    util.log.warning(f'section {sec.sid!r} in {prev.sourcePath!r} redefined in {sec.sourcePath!r}')
                self.sectionMap[sec.sid] = sec

        root = self.sectionMap.get('/')
        if not root:
            util.log.error('no root section found')
            self.sectionMap = {}
            return

        new_map = {}
        self.make_tree(root, None, new_map)

        for sec in self.sectionMap.values():
            if sec.sid not in new_map:
                util.log.warning(f'section not linked: {sec.sid!r} in {sec.sourcePath!r}')
                continue

        self.sectionMap = new_map

        for sec in self.sectionMap.values():
            self.expand_toc_nodes(sec)
            self.add_url_and_path(sec)

    def parse_file(self, path):
        return FileParser(self, path).sections()

    def make_tree(self, sec: Section, parent_sec: Section | None, new_map):
        if parent_sec:
            if sec.parentSid:
                util.log.error(f'attempt to relink section {sec.sid!r} for {sec.parentSid!r} to {parent_sec.sid!r}')
            sec.parentSid = parent_sec.sid

        if sec.status == 'ok':
            return

        if sec.status == 'walk':
            util.log.error(f'circular dependency in {sec.sid!r}')
            return

        sec.status = 'walk'

        sub_sids: list[str] = []
        new_nodes: list[ParseNode] = []
        new_map[sec.sid] = sec

        for node in sec.nodes:

            if isinstance(node, SectionNode):
                sub = self.get_section(node.sid)
                if sub:
                    self.make_tree(sub, sec, new_map)
                    sub_sids.append(sub.sid)
                    new_nodes.append(node)
                continue

            if isinstance(node, EmbedNode):
                secs = self.sections_from_wildcard_sid(node.sid, sec)
                for sub in secs:
                    self.make_tree(sub, sec, new_map)
                    sub_sids.append(sub.sid)
                    new_nodes.append(SectionNode(sid=sub.sid))
                continue

            new_nodes.append(node)

        sec.nodes = new_nodes
        sec.subSids = sub_sids
        sec.status = 'ok'

    def expand_toc_nodes(self, sec: Section):
        for node in sec.nodes:
            if isinstance(node, TocNode):
                sids = []
                for sid in node.items:
                    secs = self.sections_from_wildcard_sid(sid, sec)
                    sids.extend(s.sid for s in secs)
                node.sids = sids

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

    def cached(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]


class FileParser:
    def __init__(self, b: Builder, path):
        self.b = b
        self.path = path

    def sections(self) -> List[Section]:
        util.log.debug(f'parse {self.path!r}')

        sections = []

        dummy_root = Section(
            sid='',
            nodes=[],
            level=-1,
            headNode=MarkdownNode(el=markdown.Element(level=-1)))
        stack = [dummy_root]

        el: markdown.Element
        for el in self.parse():

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

            if el.type == 'block_code' and el.text.startswith(template.GENERATED_NODE):
                args = json.loads(el.text[len(template.GENERATED_NODE):])
                cls = globals()[args.pop('class')]
                stack[-1].nodes.append(cls(**args))
                continue

            stack[-1].nodes.append(MarkdownNode(el=el))

        return sections

    def parse(self) -> list[markdown.Element]:
        text = self.b.includeTemplate + util.read_file(self.path)
        text = template.render(self.b, text, self.path, {
            'options': self.b.options,
            'builder': self.b,
        })
        if not text:
            return []
        return self.b.markdownParser(text)

    def parse_heading(self, el: markdown.Element, parent_sec, prev_sec):
        explicit_sid = self.extract_explicit_sid(el)
        text = markdown.text_from_element(el)

        sid = self.b.make_sid(
            explicit_sid,
            parent_sec.sid,
            prev_sec.sid if prev_sec else None,
            text
        )

        if not sid and (el.level == 1 and text and not explicit_sid):
            util.log.debug(f'creating implicit root section {text!r} in {self.path!r}')
            sid = '/'

        if not sid:
            util.log.error(f'invalid section id for {text!r}:{explicit_sid!r} in {self.path!r}')
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
            status='',
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
            sec.headHtml = mr.render_children(sec.headNode.el)
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
        tpl = template.compile(self.b, self.b.options.pageTemplate)

        self.content = {}

        for path, buf in self.buffers.items():
            self.content[path] = template.call(self.b, tpl, {
                'path': path,
                'title': self.b.options.title,
                'subTitle': self.b.options.subTitle,
                'main': ''.join(buf.html),
                'breadcrumbs': self.get_breadcrumbs(buf.sids[0]),
                'builder': self.b,
                'options': self.b.options,
            })

    def write(self):
        for path, html in self.content.items():
            util.log.debug(f'write {path!r}')
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

    def link_render(self, el: markdown.Element):
        c = self.render_children(el)
        link = el.link
        if link.startswith(('http:', 'https:')):
            return self.render_a(link, el.title, c, el)
        if link.startswith('//'):
            return self.render_a(link[1:], el.title, c, el)

        sid = self.b.make_sid(link, self.sec.sid)
        target = self.b.get_section(sid)
        if not target:
            return self.render_a(link, el.title, c, el)
        return self.render_a(
            target.htmlUrl,
            el.title or target.headText,
            c or target.headHtml,
            el
        )

    def image_render(self, el: markdown.Element):
        if not el.src:
            return ''
        if el.src.startswith(('http:', 'https:')):
            return super().image_render(el)
        paths = [path for path in self.b.assetPaths if path.endswith(el.src)]
        if not paths:
            util.log.error(f'asset not found: {el.src!r} ')
            el.src = ''
            return super().image_render(el)
        el.src = self.b.add_asset(paths[0])
        return super().image_render(el)

    def heading_render(self, el: markdown.Element):
        sec = self.b.section_from_element(el)
        if not sec:
            return
        c = self.render_children(el)
        tag = 'h' + str(sec.headLevel)
        a = {'data-url': sec.htmlUrl}
        if self.b.options.debug:
            a['title'] = markdown.escape(sec.sourcePath)
        return f'<{tag}{markdown.attributes(a)}>{c}</{tag}>\n'
