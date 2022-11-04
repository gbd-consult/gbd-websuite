from typing import List


class Data:  # type: ignore
    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def __repr__(self):
        return repr(vars(self))

    def get(self, k, default=None):
        return vars(self).get(k, default)


def _data_getattr(self, attr):
    if attr.startswith('_'):
        # do not use None fallback for special props
        raise AttributeError(attr)
    return None


setattr(Data, '__getattr__', _data_getattr)


class Options(Data):
    rootDirs: List[str]
    docPatterns: List[str]
    assetPatterns: List[str]
    excludeRegex: str

    outputDir: str

    htmlSplitLevel: int
    htmlWebRoot: str
    htmlStaticDir: str
    htmlPageTemplate: str
    htmlAssets: List[str]

    serverPort: int
    serverHost: str

    title: str
    subTitle: str


class ParseNode(Data):
    pass


class MarkdownNode(ParseNode):
    align: str
    alt: str
    children: List['MarkdownNode']
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


class Section(Data):
    sid: str
    level: int

    subSids: List[str]
    parentSid: str

    sourcePath: str

    headText: str
    headHtml: str
    headNode: MarkdownNode
    headLevel: int

    nodes: List[ParseNode]

    htmlPath: str
    htmlUrl: str
    htmlBaseUrl: str
    htmlId: str

    walkColor: int


class Error(Exception):
    pass
