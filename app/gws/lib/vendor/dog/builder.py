import re
import os
import json
import fnmatch
import shutil
import tempfile
import mimetypes

from . import util as u, template, markdown, indexer
from .options import Options
from .types import (
    BaseBuilder,
    CssClasses,
    EmbedNode,
    FileBuffer,
    MarkdownElement,
    MarkdownNode,
    ParseNode,
    RawHtmlNode,
    Resource,
    Section,
    SectionNode,
    TocNode,
)


class Builder(BaseBuilder):
    markdownParser: markdown.Markdown
    htmlGenerator: 'HTMLGenerator'
    mardownGenerator: 'MarkdownGenerator'

    def __init__(self, opts: Options | dict):
        self.options = Options()
        if isinstance(opts, Options):
            opts = vars(opts)
        for k, v in opts.items():
            setattr(self.options, k, v)

        u.log.set_level('DEBUG' if self.options.debug else 'INFO')

        self.cache = {}

        self.docRoots = _check_dirs(self.options.docRoots, 'docRoots')
        self.extraAssets = _check_files(self.options.extraAssets, 'extraAssets')
        self.pageTemplate = _check_file(self.options.pageTemplate, 'pageTemplate')
        self.includeTemplate = _check_file(self.options.includeTemplate, 'includeTemplate')

        self.includeTemplateText = ''
        if self.includeTemplate:
            self.includeTemplateText = u.read_file(self.includeTemplate)

    def cached(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    ##

    def collect_and_parse(self):
        self.markdownParser = markdown.parser()

        self.docPaths = set()
        self.assetPaths = set()
        self.sectionMap = {}
        self.sectionNotFound = set()
        self.assetMap = {}
        self.jsClientSource = u.read_file(f'{os.path.dirname(__file__)}/client/{Resource.JS_CLIENT}')

        self.collect_sources()
        self.parse_all_files()

        if not self.sectionMap:
            u.log.error('no sections found')
            return False
        return True

    def build_html(self, write=False):
        if not self.collect_and_parse():
            return
        self.generate_html(write=write)
        if write:
            u.log.info(f'HTML created in {self.options.outputDir!r}')

    def build_markdown(self, write=False):
        if not self.collect_and_parse():
            return
        self.generate_markdown(write=write)
        if write:
            u.log.info(f'Markdown created in {self.options.outputDir!r}')

    def build_pdf(self):
        out_path = self.options.outputDir + '/index.pdf'
        pdf_temp_dir = tempfile.mkdtemp(prefix='dog_pdf_')

        pdf_opts = Options()
        vars(pdf_opts).update(vars(self.options))

        pdf_opts.fileSplitLevel = {'/': 0}
        pdf_opts.outputDir = pdf_temp_dir
        pdf_opts.webRoot = '.'

        if self.options.pdfPageTemplate:
            pdf_opts.pageTemplate = self.options.pdfPageTemplate

        try:
            b = Builder(pdf_opts)
            if not b.collect_and_parse():
                return
            b.generate_html(write=True)
            if self.print_to_pdf(f'{pdf_temp_dir}/index.html', out_path):
                u.log.info(f'PDF created in {out_path!r}')
        finally:
            shutil.rmtree(pdf_temp_dir, ignore_errors=True)

    def dump(self):
        def _default(x):
            d = dict(vars(x))
            d['$'] = x.__class__.__name__
            return d

        self.collect_and_parse()
        return json.dumps(
            self.sectionMap,
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            default=_default,
        )

    ##

    def collect_sources(self):
        for dirname in self.docRoots:
            self.collect_sources_from_dir(dirname)

    def collect_sources_from_dir(self, dirname):
        ex = self.options.excludeRegex

        for de in os.scandir(dirname):
            if de.name.startswith('.'):
                pass
            elif ex and re.search(ex, de.path):
                u.log.debug(f'exclude: {de.path!r}')
            elif de.is_dir():
                self.collect_sources_from_dir(de.path)
            elif de.is_file() and any(fnmatch.fnmatch(de.name, p) for p in self.options.docPatterns):
                self.docPaths.add(de.path)
            elif de.is_file() and any(fnmatch.fnmatch(de.name, p) for p in self.options.assetPatterns):
                self.assetPaths.add(de.path)

    ##

    def get_section(self, sid: str) -> Section | None:
        if sid in self.sectionNotFound:
            return
        if sid not in self.sectionMap:
            u.log.error(f'section not found: {sid!r}')
            self.sectionNotFound.add(sid)
            return
        return self.sectionMap.get(sid)

    def toc_depth(self, sid: str) -> int:
        return self.options.tocDepth.get(sid, 10**9)

    def section_by_url(self, url) -> Section | None:
        for sec in self.sectionMap.values():
            if sec.htmlBaseUrl == url:
                return sec

    def section_by_element(self, el: MarkdownElement) -> Section | None:
        for sec in self.sectionMap.values():
            if sec.headNode.el == el:
                return sec

    def sections_by_wildcard_sid(self, sid, parent_sec) -> list[Section]:
        abs_sid = self.make_sid(sid, parent_sec.sid, '', '')

        if not abs_sid:
            u.log.error(f'invalid section id {sid!r} in {parent_sec.sourcePath!r}')
            return []

        if '*' not in abs_sid:
            sub = self.get_section(abs_sid)
            if sub:
                return [sub]
            return []

        rx = '^' + '[^/]+'.join(re.escape(s) for s in abs_sid.split('*')) + '$'
        subs = [sec for sec in self.sectionMap.values() if re.match(rx, sec.sid)]
        return sorted(subs, key=lambda sec: sec.headText)

    ##

    def generate_html(self, write):
        self.assetMap = {}
        for path in self.extraAssets:
            self.register_asset_path(path)

        self.htmlGenerator = HTMLGenerator(self)
        self.htmlGenerator.render_section_heads()
        self.htmlGenerator.render_sections()
        self.htmlGenerator.flush()

        if write:
            self.htmlGenerator.write()
            self.write_assets()
            indexer.make_toc(self, save=True)
            indexer.make_index(self, save=True)
            u.write_file(
                f'{self.options.outputDir}/{self.options.staticDir}/{Resource.JS_CLIENT}',
                self.jsClientSource,
            )

    def generate_markdown(self, write):
        self.assetMap = {}
        for path in self.extraAssets:
            self.register_asset_path(path)

        self.mardownGenerator = MarkdownGenerator(self)
        self.mardownGenerator.render_section_heads()
        self.mardownGenerator.render_sections()
        self.mardownGenerator.flush()

        if write:
            self.mardownGenerator.write()
            self.write_assets()

    def print_to_pdf(self, source: str, target: str):
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
        u.ensure_dir_for(target)

        ok, _ = u.run(cmd, pipe=True)
        return ok

    ##

    def content_for_url(self, url):
        if url.endswith('.html'):
            sec = self.section_by_url(url)
            if sec:
                return 'text/html', self.htmlGenerator.buffers[sec.filePath].content
            return

        m = re.search(self.options.staticDir + '/(.+)$', url)
        if not m:
            return

        fn = m.group(1)
        if fn.endswith(Resource.JS_CLIENT):
            return 'application/javascript', self.jsClientSource
        if fn.endswith(Resource.TOC_JSON):
            return 'application/json', indexer.make_toc(self, save=False)
        if fn.endswith(Resource.INDEX_JSON) or fn.endswith(Resource.INDEX_BIN):
            idx = self.cached('_INDEX', lambda: indexer.make_index(self, save=False))
            if fn.endswith(Resource.INDEX_JSON):
                return 'application/json', idx[0]
            return 'application/octet-stream', idx[1]

        for path, fname in self.assetMap.items():
            if fname == fn:
                mt = mimetypes.guess_type(path)
                return mt[0] or 'text/plain', u.read_file_b(path)

    def register_asset_url(self, src) -> str | None:
        paths = sorted(path for path in self.assetPaths if path.endswith(src))
        if not paths:
            return None
        fname = self.register_asset_path(paths[0])
        return f'{self.options.webRoot}/{self.options.staticDir}/{fname}'

    def register_asset_path(self, path):
        if path not in self.assetMap:
            self.assetMap[path] = self.unique_asset_filename(path)
        return self.assetMap[path]

    def unique_asset_filename(self, path):
        fnames = set(self.assetMap.values())
        base, ext = os.path.splitext(os.path.basename(path))
        return u.unique_name(base, lambda name: name + ext in fnames) + ext

    def write_assets(self):
        for src, fname in self.assetMap.items():
            dst = f'{self.options.outputDir}/{self.options.staticDir}/{fname}'
            u.log.debug(f'copy {src!r} => {dst!r}')
            u.write_file_b(dst, u.read_file_b(src))

    ##

    def parse_all_files(self):
        self.sectionMap = {}

        for path in sorted(self.docPaths):
            fp = FileParser(self, path)
            for sec in fp.get_sections():
                prev = self.sectionMap.get(sec.sid)
                if prev:
                    u.log.warning(f'section redefined {sec.sid!r} from {prev.sourcePath!r} in {sec.sourcePath!r}')
                self.sectionMap[sec.sid] = sec

        root = self.sectionMap.get('/')
        if not root:
            u.log.error('no root section found')
            self.sectionMap = {}
            return

        new_map = {}
        self.make_tree(root, None, new_map)

        for sec in self.sectionMap.values():
            if sec.sid not in new_map:
                u.log.warning(f'unbound section {sec.sid!r} in {sec.sourcePath!r}')
                continue

        self.sectionMap = new_map

        for sec in self.sectionMap.values():
            self.expand_toc_nodes(sec)

        self.add_url_and_path(root, 0)

    def make_tree(self, sec: Section, parent_sec: Section | None, new_map) -> bool:
        if sec.status == 'walk':
            u.log.error(f'circular dependency in {sec.sid!r}')
            return False

        if parent_sec:
            if sec.parentSid:
                u.log.warning(f'rebinding section {sec.sid!r} from {sec.parentSid!r} to {parent_sec.sid!r}')
            sec.parentSid = parent_sec.sid

        if sec.status == 'ok':
            return True

        sec.status = 'walk'

        sub_sids: list[str] = []
        new_nodes: list[ParseNode] = []
        new_map[sec.sid] = sec

        for node in sec.nodes:
            if isinstance(node, SectionNode):
                sub = self.get_section(node.sid)
                if sub and self.make_tree(sub, sec, new_map):
                    sub_sids.append(sub.sid)
                    new_nodes.append(node)
                continue

            if isinstance(node, EmbedNode):
                secs = self.sections_by_wildcard_sid(node.sid, sec)
                for sub in secs:
                    if self.make_tree(sub, sec, new_map):
                        sub_sids.append(sub.sid)
                        new_nodes.append(SectionNode(sid=sub.sid))
                continue

            new_nodes.append(node)

        sec.nodes = new_nodes
        sec.subSids = sub_sids
        sec.status = 'ok'
        return True

    def expand_toc_nodes(self, sec: Section):
        for node in sec.nodes:
            if isinstance(node, TocNode):
                sids = []
                for sid in node.items:
                    secs = self.sections_by_wildcard_sid(sid, sec)
                    sids.extend(s.sid for s in secs)
                node.sids = sids

    def add_url_and_path(self, sec: Section, split_level):
        if sec.sid in self.options.fileSplitLevel:
            split_level = self.options.fileSplitLevel[sec.sid]

        parts = sec.sid.split('/')[1:]

        if sec.level == 0 or split_level == 0:
            path = 'index.html'
        else:
            dirname = '/'.join(parts[:split_level])
            path = dirname + '/index.html'

        sec.filePath = f'{self.options.outputDir}/{path}'
        sec.htmlBaseUrl = f'{self.options.webRoot}/{path}'
        sec.htmlId = '-'.join(parts[split_level:])

        u.log.debug(f'path {sec.sid} -> {sec.filePath} ({split_level})')

        sec.htmlUrl = sec.htmlBaseUrl
        if sec.htmlId:
            sec.htmlUrl += '#' + sec.htmlId

        sec.headLevel = max(1, sec.level - split_level + 1)

        for sub in sec.subSids:
            sub = self.sectionMap[sub]
            self.add_url_and_path(sub, split_level)

    def make_sid(self, explicit_sid, parent_sid, prev_sid=None, text=None):
        explicit_sid = explicit_sid or ''
        text_sid = u.to_uid(text) if text else ''

        if explicit_sid == '/':
            return '/'

        sid = explicit_sid or text_sid
        if sid.endswith('/'):
            sid += text_sid
        if not sid or sid.endswith('/'):
            return ''

        if sid.startswith('/'):
            return u.normpath(sid)

        if parent_sid:
            return u.normpath(parent_sid + '/' + sid)

        if prev_sid:
            ps, _, _ = prev_sid.rpartition('/')
            return u.normpath(ps + '/' + sid)

        return ''


class FileParser:
    def __init__(self, b: Builder, path):
        self.b = b
        self.path = path

    def get_sections(self) -> list[Section]:
        u.log.debug(f'parse {self.path!r}')

        sections = []

        dummy_root = Section(
            sid='',
            nodes=[],
            level=-1,
            headNode=MarkdownNode(el=MarkdownElement(type='', level=-1)),
        )
        stack = [dummy_root]

        text = self.render_as_template()
        if not text:
            return []

        for el in self.b.markdownParser(text):
            if el.type == 'heading':
                if self.strip_plain_marker(el):
                    stack[-1].nodes.append(MarkdownNode(el=el))
                    continue

                prev_sec = None
                while stack[-1].headNode.el.level > el.level:
                    stack.pop()
                if stack[-1].headNode.el.level == el.level:
                    prev_sec = stack.pop()

                sec = self.make_section(el, stack[-1], prev_sec)
                if sec:
                    stack.append(sec)
                    sections.append(sec)

                continue

            if el.type == 'block_code' and el.text.startswith(template.GENERATED_NODE):
                args = json.loads(el.text[len(template.GENERATED_NODE) :])
                cls = globals()[args.pop('class')]
                stack[-1].nodes.append(cls(**args))
                continue

            stack[-1].nodes.append(MarkdownNode(el=el))

        return sections

    def render_as_template(self):
        text = self.b.includeTemplateText + u.read_file(self.path)
        return template.render(
            self.b,
            text,
            self.path,
            {
                'options': self.b.options,
                'builder': self.b,
            },
        )

    def make_section(self, el: MarkdownElement, parent_sec, prev_sec):
        explicit_sid = self.extract_explicit_sid(el)
        text = markdown.text_from_element(el)

        sid = self.b.make_sid(explicit_sid, parent_sec.sid, prev_sec.sid if prev_sec else None, text)

        if not sid and (el.level == 1 and text and not explicit_sid):
            u.log.debug(f'creating implicit root section {text!r} in {self.path!r}')
            sid = '/'

        if not sid:
            u.log.error(f'invalid section id for {text!r}:{explicit_sid!r} in {self.path!r}')
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

    def extract_explicit_sid(self, el: MarkdownElement) -> str:
        ch = el.children

        if not ch or ch[-1].type != 'text':
            return ''

        m = re.match(r'^(?:(.*?)\s+)?:([\w./*-]+)$', ch[-1].text)
        if not m:
            return ''

        ch[-1].text = m.group(1) or ''
        markdown.strip_text_content(el)

        return m.group(2)

    def strip_plain_marker(self, el: MarkdownElement) -> bool:
        ch = el.children

        if not ch or ch[-1].type != 'text':
            return False

        m = re.match(r'^(?:(.*?)\s+)?::$', ch[-1].text)
        if not m:
            return False

        ch[-1].text = m.group(1) or ''
        markdown.strip_text_content(el)
        el.isPlainHeading = True
        el.htmlId = u.to_uid(markdown.text_from_element(el))

        return True


class HTMLGenerator:
    def __init__(self, b: Builder):
        self.b = b
        self.buffers: dict[str, FileBuffer] = {}

    def render_section_heads(self):
        for sec in self.b.sectionMap.values():
            mr = HTMLRenderer(self.b, sec)
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

        u.log.debug(f'render {sid!r}')

        mr = HTMLRenderer(self.b, sec)

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
                html = f'<div class="{CssClasses.LOCAL_TOC}"><ul>{entries}</ul></div>'
                self.add(sec, html)
                continue
            if isinstance(node, RawHtmlNode):
                self.add(sec, node.html)
                continue

        self.add(sec, f'</section>\n')

    def render_toc_entry(self, sid, depth: int):
        sec = self.b.get_section(sid)
        if not sec:
            return ''

        depth = min(depth, self.b.toc_depth(sid))
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
        return '\n'.join(self.render_toc_entry(sid, 999) for sid in root.subSids)

    def add(self, sec: Section, chunk: str):
        if sec.filePath not in self.buffers:
            self.buffers[sec.filePath] = FileBuffer(sids=[], chunks=[], content='')
        self.buffers[sec.filePath].sids.append(sec.sid)
        self.buffers[sec.filePath].chunks.append(chunk)

    def flush(self):
        if not self.b.pageTemplate:
            return

        tpl = template.compile(self.b, self.b.pageTemplate)
        if not tpl:
            return

        home_url = ''
        sec = self.b.get_section('/')
        if sec:
            home_url = sec.htmlUrl

        for path, buf in self.buffers.items():
            buf.content = template.call(
                self.b,
                tpl,
                {
                    'path': path,
                    'main': ''.join(buf.chunks),
                    'breadcrumbs': self.get_breadcrumbs(buf.sids[0]),
                    'home': home_url,
                    'builder': self.b,
                    'options': self.b.options,
                    'args': self.b.options.pageTemplateArgs or {},
                },
            )

    def write(self):
        for path, buf in self.buffers.items():
            u.log.debug(f'write {path!r}')
            u.write_file(path, buf.content)

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


class HTMLRenderer(markdown.HTMLRenderer):
    def __init__(self, b: Builder, sec: Section):
        self.b = b
        self.sec = sec

    def r_link(self, el: MarkdownElement):
        c = self.render_children(el)
        if el.target.startswith(('http:', 'https:')):
            return self.render_link(el.target, el.title, c, el)
        if el.target.startswith('//'):
            return self.render_link(el.target[1:], el.title, c, el)

        url = self.b.register_asset_url(el.target)
        if url:
            return self.render_link(url, el.title, c, el)

        sid = self.b.make_sid(el.target, self.sec.sid)
        sec = self.b.get_section(sid)
        if not sec:
            return self.render_link(el.target, el.title, c, el)
        return self.render_link(sec.htmlUrl, el.title or sec.headText, c or sec.headHtml, el)

    def r_image(self, el: MarkdownElement):
        if not el.src:
            return ''
        if el.src.startswith(('http:', 'https:')):
            return super().r_image(el)
        url = self.b.register_asset_url(el.src)
        if not url:
            u.log.error(f'asset not found: {el.src!r} ')
            el.src = ''
            return super().r_image(el)
        el.src = url
        return super().r_image(el)

    def r_heading(self, el: MarkdownElement):
        if el.isPlainHeading:
            c = self.render_children(el)
            tag = 'h' + str(min(6, max(1, el.level)))
            a = {}
            if el.htmlId:
                a['id'] = el.htmlId
                a['data-url'] = self.sec.htmlBaseUrl + '#' + el.htmlId
            return f'<{tag}{markdown.attributes(a)}>{c}</{tag}>\n'
        sec = self.b.section_by_element(el)
        if not sec:
            return super().r_heading(el)
        c = self.render_children(el)
        tag = 'h' + str(sec.headLevel)
        a = {'data-url': sec.htmlUrl}
        if self.b.options.debug:
            a['title'] = markdown.escape(sec.sourcePath)
        return f'<{tag}{markdown.attributes(a)}>{c}</{tag}>\n'


class MarkdownGenerator:
    def __init__(self, b: Builder):
        self.b = b
        self.buffers: dict[str, FileBuffer] = {}

    def render_section_heads(self):
        for sec in self.b.sectionMap.values():
            mr = MarkdownRenderer(self.b, sec)
            sec.headHtml = mr.render_children(sec.headNode.el)
            sec.headHtmlLink = f'[{sec.headHtml}]({sec.htmlUrl})'

    def render_sections(self):
        for sec in self.b.sectionMap.values():
            if not sec.parentSid:
                self.render_section(sec.sid)

    def render_section(self, sid):
        sec = self.b.get_section(sid)
        if not sec:
            return

        u.log.debug(f'render {sid!r}')

        mr = MarkdownRenderer(self.b, sec)

        for node in sec.nodes:
            if isinstance(node, MarkdownNode):
                text = mr.render_element(node.el)
                self.add(sec, text)
                continue
            if isinstance(node, SectionNode):
                self.render_section(node.sid)
                continue
            if isinstance(node, TocNode):
                entries = ''.join(self.render_toc_entry(sid, node.depth, 0) for sid in node.sids)
                self.add(sec, entries + '\n')
                continue
            if isinstance(node, RawHtmlNode):
                self.add(sec, node.html)
                continue

    def render_toc_entry(self, sid, depth: int, level: int):
        sec = self.b.get_section(sid)
        if not sec:
            return ''

        depth = min(depth, self.b.toc_depth(sid))
        s = ('  ' * level) + '- ' + sec.headHtmlLink + '\n'
        if depth > 1:
            s += ''.join(self.render_toc_entry(s2, depth - 1, level + 1) for s2 in sec.subSids)

        return s

    def add(self, sec: Section, chunk: str):
        if sec.filePath not in self.buffers:
            self.buffers[sec.filePath] = FileBuffer(sids=[], chunks=[], content='')
        self.buffers[sec.filePath].sids.append(sec.sid)
        self.buffers[sec.filePath].chunks.append(chunk)

    def flush(self):
        for path, buf in self.buffers.items():
            buf.content = ''.join(buf.chunks)

    def write(self):
        for path, buf in self.buffers.items():
            path = path.replace('.html', '.md')
            u.log.debug(f'write {path!r}')
            u.write_file(path, buf.content)


class MarkdownRenderer(markdown.MarkdownRenderer):
    def __init__(self, b: Builder, sec: Section):
        self.b = b
        self.sec = sec

    def r_link(self, el: MarkdownElement):
        c = self.render_children(el)
        if el.target.startswith(('http:', 'https:')):
            return self.render_link(el.target, el.title, c, el)
        if el.target.startswith('//'):
            return self.render_link(el.target[1:], el.title, c, el)

        url = self.b.register_asset_url(el.target)
        if url:
            return self.render_link(url, el.title, c, el)

        sid = self.b.make_sid(el.target, self.sec.sid)
        sec = self.b.get_section(sid)
        if not sec:
            return self.render_link(el.target, el.title, c, el)
        return self.render_link(sec.htmlUrl, el.title or sec.headText, c or sec.headHtml, el)

    def r_image(self, el: MarkdownElement):
        if not el.src:
            return ''
        if el.src.startswith(('http:', 'https:')):
            return super().r_image(el)
        url = self.b.register_asset_url(el.src)
        if not url:
            u.log.error(f'asset not found: {el.src!r} ')
            el.src = ''
            return super().r_image(el)
        el.src = url
        return super().r_image(el)

    def r_heading(self, el: MarkdownElement):
        sec = self.b.section_by_element(el)
        if not sec:
            return super().r_heading(el)
        c = self.render_children(el)
        return ('#' * sec.headLevel) + ' ' + c + '\n\n'


##


def _check_dirs(paths, name):
    res = []
    for p in paths or []:
        p = os.path.abspath(p)
        if not os.path.isdir(p):
            u.log.error(f'{name}: directory not found: {p!r}')
            continue
        if p not in res:
            res.append(p)
    return res


def _check_files(paths, name):
    res = []
    for p in paths or []:
        p = _check_file(p, name)
        if p and p not in res:
            res.append(p)
    return res


def _check_file(path, name):
    if not path:
        return ''
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        u.log.error(f'{name}: file not found: {path!r}')
        return ''
    return path
