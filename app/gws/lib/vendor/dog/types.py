from .util import Data
from .options import Options


class MarkdownElement(Data):
    type: str

    align: str
    alt: str
    children: list['MarkdownElement']
    info: str
    isTableHead: bool
    isPlainHeading: bool
    htmlId: str
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


class ParseNode(Data):
    pass


class MarkdownNode(ParseNode):
    el: MarkdownElement


class SectionNode(ParseNode):
    sid: str


class EmbedNode(ParseNode):
    items: list[str]
    sid: str


class TocNode(ParseNode):
    items: list[str]
    sids: list[str]
    depth: int


class RawHtmlNode(ParseNode):
    html: str


class Section(Data):
    sid: str
    level: int
    status: str
    subSids: list[str]
    parentSid: str

    sourcePath: str

    headText: str
    headHtml: str
    headHtmlLink: str
    headNode: MarkdownNode
    headLevel: int

    nodes: list[ParseNode]

    filePath: str
    htmlUrl: str
    htmlBaseUrl: str
    htmlId: str


class FileBuffer(Data):
    sids: list[str]
    chunks: list[str]
    content: str


class BaseBuilder:
    options: Options
    includeTemplateText: str
    cache: dict

    docRoots: list[str]
    extraAssets: list[str]
    pageTemplate: str
    includeTemplate: str

    docPaths: set[str]
    assetPaths: set[str]
    sectionMap: dict[str, Section]
    sectionNotFound: set[str]
    assetMap: dict[str, str]

    def cached(self, key, fn):
        raise NotImplementedError


class Resource:
    TOC_JSON = '_toc.json'
    INDEX_JSON = '_index.json'
    INDEX_BIN = '_index.bin'
    JS_CLIENT = '_dog.js'


class CssClasses:
    CODE_BLOCK = 'md-code-block'
    CODE_BLOCK_TITLE = 'md-code-block-title'
    ERROR = 'md-error'
    TABLE = 'md-table'
    DECORATION = 'md-decoration-'
    LOCAL_TOC = 'md-local-toc'
    ADMONITION_INFO = 'md-admonition-info'
    ADMONITION_WARN = 'md-admonition-warn'
    ADMONITION_SEE = 'md-admonition-see'

