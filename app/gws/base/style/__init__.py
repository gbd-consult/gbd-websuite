import gws.types as t
import gws.lib.style


#:export
class StyleType(t.Enum):
    css = 'css'
    cssSelector = 'cssSelector'


class Config(t.Config):
    """Feature style"""

    type: t.StyleType  #: style type
    name: t.Optional[str]  #: style name
    text: t.Optional[str]  #: raw style content
    values: t.Optional[dict]  #: style values


#:export
class StyleProps(t.Props):
    type: t.StyleType
    name: t.Optional[str]
    text: t.Optional[str]
    values: t.Optional[t.StyleValues]


def from_props(p: t.StyleProps) -> t.IStyle:
    if p.type == 'css':
        if p.values:
            values = gws.lib.style.from_css_dict(gws.as_dict(p.values))
        else:
            values = gws.lib.style.from_css_text(p.text)
        return Style(p.type, values=values, name=gws.get(p, 'name'))

    if p.type == 'cssSelector':
        return Style(p.type, name=(p.name or ''))

    raise gws.Error(f'invalid style type {p.type!r}')


def from_config(c: Config) -> t.IStyle:
    return from_props(t.cast(StyleProps, c))


#:export IStyle
class Style(t.IStyle):
    def __init__(self, type: StyleType, values: t.StyleValues = None, text: str = None, name: str = None):
        super().__init__()
        self.type: StyleType = type
        self.values: t.StyleValues = values
        self.text: str = text
        self.name: str = name

    @property
    def props(self) -> t.StyleProps:
        return t.StyleProps(
            type=self.type,
            values=self.values,
            text=self.text or '',
            name=self.name or '')
