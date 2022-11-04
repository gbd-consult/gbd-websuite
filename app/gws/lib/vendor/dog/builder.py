import re
import os
import json
import fnmatch
import shutil
import mimetypes

import jump

from . import types as t
from . import util, markdown


class JumpEngine(jump.Engine):
    options = dict(
        echo_open_symbol='<%',
        echo_close_symbol='%>',
        echo_start_whitespace=True,
    )

    COMMAND = '__DG__'

    @staticmethod
    def error(exc, source_path, source_lineno, env):
        util.log.error(f'template error: {exc.args[0]} in {source_path!r}')

    def emit_code_block(self, command, args):
        args['command'] = command
        js = json.dumps(args)
        return f'```\n{self.COMMAND}{js}\n```\n'

    def box_toc(self, text, depth=1):
        items = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        return self.emit_code_block('TocNode', {'items': items, 'depth': depth})


class SectionNode(t.ParseNode):
    sid: str


class InsertSectionNode(t.ParseNode):
    items: t.List[str]
    sid: str


class TocNode(t.ParseNode):
    items: t.List[str]
    sids: t.List[str]
    depth: int


class Builder:
    options: t.Options

    def __init__(self, options, debug=False):
        self.options = options

        self.docPaths = []
        self.assetPaths = []
        self.assetsUsed = []

        self.markdownParser = markdown.parser()
        self.jumpEngine = JumpEngine()

        self.sectionMap: dict = {}

        self.htmlBuffers: dict = {}
        self.htmlContent: dict = {}

        self.debug = debug

    def build_all(self, mode: str, write=False):
        util.log.debug(f'START build_all {mode}')

        os.makedirs(self.options.outputDir, exist_ok=True)

        self.htmlBuffers = {}

        self.sectionMap = {sec.sid: sec for sec in self.parse_all()}

        for sec in self.sectionMap.values():
            self.make_tree(sec)

        for sec in self.sectionMap.values():
            self.add_url_and_path(sec)

        self.dump({'sec': list(self.sectionMap.values())})

        if mode == 'html':
            for sec in self.sectionMap.values():
                if not sec.parentSid:
                    self.html_render_section(sec.sid)
            self.html_flush(write)

        util.log.debug(f'END build_all {mode}')

    def parse_all(self):
        self.collect_sources()

        for path in self.options.htmlAssets:
            self.assetPaths.append(path)
            self.assetsUsed.append(path)

        return [
            sec
            for path in self.docPaths
            for sec in self.parse_file(path)
        ]

    def parse_file(self, path):
        return FileParser(self, path).parse()

    def collect_sources(self):
        for root in self.options.rootDirs:
            self.collect_sources_from_dir(root)

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

    def get_section(self, sid) -> t.Section:
        if sid not in self.sectionMap:
            util.log.debug(f'section {sid!r} not found')
        return self.sectionMap.get(sid)

    def section_from_url(self, url):
        for sec in self.sectionMap.values():
            if sec.htmlBaseUrl == url:
                return sec

    def sections_from_wildcard_sid(self, sid, parent_sec):
        abs_sid = self.make_sid(sid, parent_sec.sid, '', '')

        if not abs_sid:
            util.log.error(f'invalid section id {sid!r} in {parent_sec.sourcePath!r}')
            return []

        if '*' not in abs_sid:
            sub = self.get_section(abs_sid)
            if sub:
                return [sub]
            util.log.error(f'invalid section id {sid!r} in {parent_sec.sourcePath!r}')
            return []

        rx = abs_sid.replace('*', '[^/]+') + '$'
        subs = [
            sec
            for sec in self.sectionMap.values()
            if re.match(rx, sec.sid)
        ]
        return sorted(subs, key=lambda sec: sec.headText)

    ##

    def content_for_url(self, url):
        if url.endswith('.html'):
            sec = self.section_from_url(url)
            if sec:
                return 'text/html', self.htmlContent[sec.htmlPath]
            return

        m = re.search(self.options.htmlStaticDir + '/(.+)$', url)
        if m:
            fname = m.group(1)
            for path in self.assetPaths:
                if os.path.basename(path) == fname:
                    mt = mimetypes.guess_type(path)
                    mime = mt[0] if mt else 'text/plain'
                    return mime, util.read_file_b(path)

    ##

    def html_render_section(self, sid):
        util.log.debug(f'render {sid!r}')
        sec = self.get_section(sid)
        if not sec:
            util.log.error(f'section {sid!r} not found')
            return
        r = HTMLRenderer(self, sec)
        r.render_section()

    def html_render_toc_entry(self, sid, depth: int):
        sec = self.get_section(sid)
        if not sec:
            util.log.error(f'section {sid!r} not found')
            return ''

        r = HTMLRenderer(self, sec)
        c = r.render_content(sec.headNode)
        a = f'<a href="{sec.htmlUrl}">{c}</a>'
        s = ''

        if depth > 1:
            sub = [self.html_render_toc_entry(s, depth - 1) for s in sec.subSids]
            if sub:
                s = '<ul>' + ''.join(sub) + '</ul>'

        return f'<li data-sid="{sid}">{a}{s}</li>'

    def html_render_main_toc(self):
        root = self.get_section('/')
        if not root:
            return ''
        entries = [
            self.html_render_toc_entry(sid, 999)
            for sid in root.subSids
        ]
        return '\n'.join(entries)

    def html_write(self, sec: t.Section, html):
        self.htmlBuffers.setdefault(sec.htmlPath, []).append(html)

    def html_flush(self, write: bool):
        tpl = self.jumpEngine.compile_path(self.options.htmlPageTemplate, **JumpEngine.options)
        maintoc = self.html_render_main_toc()

        self.htmlContent = {}

        for path, fragments in self.htmlBuffers.items():
            self.htmlContent[path] = self.jumpEngine.call(tpl, {
                'title': self.options.title,
                'subTitle': self.options.subTitle,
                'maintoc': maintoc,
                'main': ''.join(fragments),
                'builder': self,
                'options': self.options,
            })

        if not write:
            return

        if os.path.isdir(self.options.outputDir):
            shutil.rmtree(self.options.outputDir)

        for path, html in self.htmlContent.items():
            util.log.debug(f'write {path!r}')
            dirs = os.path.dirname(path)
            os.makedirs(dirs, exist_ok=True)
            util.write_file(path, html)

        dirs = self.options.outputDir + '/' + self.options.htmlStaticDir
        os.makedirs(dirs, exist_ok=True)
        for src in self.assetsUsed:
            dst = dirs + '/' + src.split('/').pop()
            util.log.debug(f'copy {src!r} => {dst!r}')
            shutil.copy(src, dst)

    ##

    def make_tree(self, sec: t.Section, parent_sec=None):
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
                if not sub:
                    continue
                self.make_tree(sub, sec)
                sec.subSids.append(sub.sid)
                sec.nodes.append(node)
                continue

            if isinstance(node, InsertSectionNode):
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

    def add_url_and_path(self, sec: t.Section):
        sl = self.options.htmlSplitLevel or 0
        parts = sec.sid.split('/')[1:]

        if sec.level == 0 or sl == 0:
            path = 'index.html'
        else:
            dirname = '/'.join(parts[:sl])
            path = dirname + '/index.html'

        sec.htmlId = '-'.join(parts[sl:])
        sec.htmlPath = self.options.outputDir + '/' + path
        sec.htmlBaseUrl = self.options.htmlWebRoot + '/' + path

        sec.htmlUrl = sec.htmlBaseUrl
        if sec.htmlId:
            sec.htmlUrl += '#' + sec.htmlId

        sec.headLevel = max(1, sec.level - sl + 1)

    ##

    def make_sid(self, explicit_sid, parent_sid, prev_sid=None, text=None):

        """
            ex.sid      parent_sid   prev_sid   result
            ---------------------------------------------
            /           -            -          /
            /abs        -            -          /abs   
            /abs/       -            -          /abs/text   
           
            rel         y            -          parent_sec-sid/rel
            rel/        y            -          parent_sec-sid/rel/text
            none        y            -          parent_sec-sid/text
           
            rel         n            y          parent_sec-sid-without-last-comp/rel
            rel/        n            y          parent_sec-sid-without-last-comp/rel/text
            none        n            y          parent_sec-sid-without-last-comp/text

        """

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
        self.dummyRoot = t.Section(sid='', nodes=[], level=-1, headNode=t.MarkdownNode(level=-1))

    def parse(self) -> t.List[t.Section]:
        util.log.debug(f'parse {self.path!r}')

        src = self.pre_parse()
        if not src:
            return []

        sections = []
        stack = [self.dummyRoot]

        node: t.MarkdownNode
        for node in self.b.markdownParser(src):

            if node.type == 'heading':
                prev_sec = None
                while stack[-1].headNode.level > node.level:
                    stack.pop()
                if stack[-1].headNode.level == node.level:
                    prev_sec = stack.pop()

                sec = self.parse_heading(node, stack[-1], prev_sec)
                if sec:
                    stack.append(sec)
                    sections.append(sec)

                continue

            if node.type == 'block_code' and node.text.startswith(JumpEngine.COMMAND):
                args = json.loads(node.text[len(JumpEngine.COMMAND):])
                cls = globals()[args['command']]
                stack[-1].nodes.append(cls(**args))
                continue

            stack[-1].nodes.append(node)

        return sections

    def pre_parse(self):

        try:
            return self.b.jumpEngine.render_path(
                self.path,
                error=JumpEngine.error,
                **JumpEngine.options,
            )
        except jump.CompileError as exc:
            util.log.error(f'compile error {exc.args[0]}')

    def parse_heading(self, node, parent_sec, prev_sec):
        explicit_sid = self.extract_explicit_sid(node)
        text = markdown.text_from_node(node)

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
            parent_sec.nodes.append(InsertSectionNode(sid=sid))
            return

        parent_sec.nodes.append(SectionNode(sid=sid))
        node.sid = sid

        return t.Section(
            sid=sid,
            level=0 if sid == '/' else sid.count('/'),
            sourcePath=self.path,
            headText=text,
            headNode=node,
            nodes=[node],
        )

    def extract_explicit_sid(self, node: t.MarkdownNode) -> str:
        ch = node.children

        if not ch or ch[-1].type != 'text':
            return ''

        m = re.match(r'^(.*?):(\S+)$', ch[-1].text)
        if not m:
            return ''

        ch[-1].text = m.group(1)
        markdown.strip_node(node)

        return m.group(2)


class HTMLRenderer(markdown.HTMLRenderer):

    def __init__(self, b: Builder, sec: t.Section):
        self.b = b
        self.sec = sec

    def render_section(self):
        self.b.html_write(self.sec, f'<section id="{self.sec.htmlId}" data-sid="{self.sec.sid}">\n')

        for node in self.sec.nodes:
            if isinstance(node, t.MarkdownNode):
                html = self.render_node(node)
                self.b.html_write(self.sec, html)
                continue
            if isinstance(node, SectionNode):
                self.b.html_render_section(node.sid)
                continue
            if isinstance(node, TocNode):
                entries = ''.join(self.b.html_render_toc_entry(sid, node.depth) for sid in node.sids)
                html = f'<div class="localtoc"><ul>{entries}</ul></div>'
                self.b.html_write(self.sec, html)
                continue

        self.b.html_write(self.sec, f'</section>\n')

    ##

    def tag_link(self, node):
        c = self.render_content(node)
        link = node.link
        if link.startswith(('http:', 'https:')):
            return self.render_a(link, node.title, c)
        if link.startswith('//'):
            return self.render_a(link[1:], node.title, c)

        sid = self.b.make_sid(link, self.sec.sid)
        target = self.b.get_section(sid)
        if not target:
            util.log.error(f'invalid link {link!r} in {self.sec.sourcePath!r}')
            return self.render_a(link, node.title, c)

        return self.render_a(
            target.htmlUrl,
            node.title or target.headText,
            c or self.render_content(target.headNode))

    def tag_heading(self, node):
        c = self.render_content(node)
        sec = self.b.get_section(node.sid)
        tag = 'h' + str(sec.headLevel)
        s = ''
        if self.b.debug:
            s = f' title="{markdown.escape(sec.sourcePath)}"'
        return f'<{tag}{s}>{c}</{tag}>\n'