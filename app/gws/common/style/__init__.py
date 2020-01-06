import gws.types as t


class Config(t.Config):
    """Feature style"""

    type: str  #: style type ("css")
    content: t.Optional[dict]  #: css rules
    text: t.Optional[str]  #: raw style content


def from_props(p: t.StyleProps) -> t.IStyle:
    if p.type == 'css':
        content = p.content or _parse_css(p.text or '')
        s: t.IStyle = Style('css', content)
        return s


def from_config(c: Config) -> t.IStyle:
    p: t.StyleProps = c
    return from_props(p)


#:export
class StyleProps(t.Props):
    type: str
    content: t.Optional[dict]
    text: t.Optional[str]


class Config(t.Config):
    """Feature style"""

    type: str  #: style type ("css")
    content: t.Optional[dict]  #: css rules
    text: t.Optional[str]  #: raw style content


#:export IStyle
class Style(t.IStyle):
    def __init__(self, type, content):
        super().__init__()
        self.type: str = type
        self.content: dict = content

    @property
    def text(self) -> str:
        if self.type == 'css':
            return _make_css(self.content)

    @property
    def props(self) -> t.StyleProps:
        return t.StyleProps({
            'type': self.type,
            'content': self.content
        })


# @TODO use a real parser
def _parse_css(text):
    content = {}
    for r in text.split(';'):
        r = r.strip()
        if not r:
            continue
        r = r.split(':')
        content[r[0].strip()] = r[1].strip()
    return content


def _make_css(content):
    return ';'.join(f'{k}: {v}' for k, v in content.items())
