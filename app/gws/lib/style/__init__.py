import gws
import gws.types as t

from . import parser, icon

##

# parsing depends on whenever the context is `trusted` (=config) or not (=request)

def from_dict(d: dict, trusted=False, with_strict_mode=True) -> 'Style':
    vals = {}

    s = d.get('text')
    if s:
        vals.update(parser.parse_text(s, trusted, with_strict_mode))

    s = d.get('values')
    if s:
        vals.update(parser.parse_dict(gws.to_dict(s), trusted, with_strict_mode))

    return Style(
        name=d.get('name', ''),
        values=gws.StyleValues(vals),
        selector=d.get('selector', ''),
        text=d.get('text', ''),
    )


def from_args(**kwargs) -> 'Style':
    d = {'values': kwargs}
    return from_dict(d, trusted=True, with_strict_mode=True)


def from_config(cfg: gws.Config) -> 'Style':
    d = gws.to_dict(cfg)
    return from_dict(d, trusted=True, with_strict_mode=True)


def from_props(props: gws.Props) -> 'Style':
    d = gws.to_dict(props)
    return from_dict(d, trusted=False, with_strict_mode=False)


def from_text(text: str) -> 'Style':
    d = {'text': text}
    return from_dict(d, trusted=True, with_strict_mode=True)


##
class Props(gws.Props):
    name: t.Optional[str]
    values: t.Optional[dict]
    selector: t.Optional[str]
    text: t.Optional[str]


class Config(gws.Config):
    """Feature style"""

    name: t.Optional[str] 
    """style name"""
    selector: t.Optional[str] 
    """CSS selector"""
    text: t.Optional[str] 
    """raw style content"""
    values: t.Optional[dict] 
    """style values"""


class Style(gws.Object, gws.IStyle):
    def __init__(self, name, selector, text, values):
        super().__init__()
        self.name = name
        self.selector = selector
        self.text = text
        self.values = values

    def props(self, user):
        ico = self.values.icon
        if ico and isinstance(ico, icon.ParsedIcon):
            ico = icon.to_data_url(ico)
        else:
            # NB if icon is not parsed, don't give it back
            ico = ''

        return gws.Data(
            values=gws.merge(self.values, icon=ico),
            name=self.name or '',
            selector=self.selector or '',
        )
