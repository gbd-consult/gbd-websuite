import gws.types as t
import gws.tools.style


#:export
class StyleType(t.Enum):
    css = 'css'
    cssSelector = 'cssSelector'


class Config(t.Config):
    """Feature style"""

    type: t.StyleType  #: style type
    text: str  #: raw style content / selector


#:export
class StyleProps(t.Props):
    type: t.StyleType
    values: t.Optional[t.StyleValues]
    text: t.Optional[str]


def from_props(p: t.StyleProps) -> t.IStyle:
    if p.type == 'css':
        values = p.values or gws.tools.style.from_css_text(p.text)
        s: t.IStyle = Style(p.type, values=values)
        return s

    if p.type == 'cssSelector':
        s: t.IStyle = Style(p.type, text=(p.text or ''))
        return s

    raise ValueError(f'invalid style type {p.type!r}')


def from_config(c: Config) -> t.IStyle:
    return from_props(t.StyleProps(type=c.type, text=c.text))


#:export IStyle
class Style(t.IStyle):
    def __init__(self, type: StyleType, values: t.StyleValues = None, text: str = None):
        super().__init__()
        self.type: StyleType = type
        self.values: t.StyleValues = values
        self.text: str = text

    @property
    def props(self) -> t.StyleProps:
        return t.StyleProps(
            type=self.type,
            values=self.values.as_dict() if self.values else None,
            text=self.text or '')
