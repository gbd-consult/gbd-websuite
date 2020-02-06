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


def from_props(root: t.IRootObject, p: t.StyleProps) -> t.IStyle:
    # @TODO: icon urls are resolved from the _first_ configured site

    site: t.IWebSite = root.find_first('gws.web.site')
    url_root = site.static_root.dir if site else None

    if p.type == 'css':
        if p.values:
            values = gws.tools.style.from_css_dict(gws.as_dict(p.values), url_root)
        else:
            values = gws.tools.style.from_css_text(p.text, url_root)
        return Style(p.type, values=values)

    if p.type == 'cssSelector':
        return Style(p.type, text=(p.text or ''))

    raise gws.Error(f'invalid style type {p.type!r}')


def from_config(root: t.IRootObject, c: Config) -> t.IStyle:
    return from_props(root, t.StyleProps(type=c.type, text=c.text))


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
