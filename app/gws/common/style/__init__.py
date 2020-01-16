import gws.types as t

#:export
class StyleType(t.Enum):
    css = 'css'
    cssSelector = 'cssSelector'


class Config(t.Config):
    """Feature style"""

    type: t.StyleType  #: style type
    text: t.Optional[str]  #: raw style content / selector


def from_props(p: t.StyleProps) -> t.IStyle:
    if p.type == 'css':
        content = p.content or _parse_css(p.text or '')
        s: t.IStyle = Style(p.type, content=content)
        return s
    if p.type == 'cssSelector':
        s: t.IStyle = Style(p.type, text=(p.text or ''))
        return s
    raise ValueError(f'invalid style type {p.type!r}')


def from_config(c: Config) -> t.IStyle:
    return from_props(t.StyleProps(type=c.type, text=c.text))


#:export
class StyleProps(t.Props):
    type: t.StyleType
    content: dict
    text: str


#:export IStyle
class Style(t.IStyle):
    def __init__(self, type, content=None, text=None):
        super().__init__()
        self.type: str = type
        self.content: dict = content
        self.text = text

    @property
    def props(self) -> t.StyleProps:
        return t.StyleProps(
            type=self.type,
            content=self.content or {},
            text=self.text or '')


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
