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
        if p.values:
            values = gws.tools.style.from_css_dict(gws.as_dict(p.values))
        else:
            values = gws.tools.style.from_css_text(p.text)
        return Style(p.type, values=values)

    if p.type == 'cssSelector':
        return Style(p.type, text=(p.text or ''))

    raise gws.Error(f'invalid style type {p.type!r}')


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
            values=self.values,
            text=self.text or '')
