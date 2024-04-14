import gws
import gws.types as t

from . import parser, icon


# parsing depends on whenever the context is `trusted` (=config) or not (=request)

def from_dict(d: dict, opts: parser.Options = None) -> 'Object':
    """Creates a css style object from a dictionary.

    Args:
        d: A dictionary.
        opts: Dictionary options.

    Returns:
        A css style object.

    """
    vals = {}
    opts = opts or parser.Options(trusted=False, strict=True)

    s = d.get('text')
    if s:
        vals.update(parser.parse_text(s, opts))

    s = d.get('values')
    if s:
        vals.update(parser.parse_dict(gws.u.to_dict(s), opts))

    return Object(
        d.get('cssSelector', ''),
        d.get('text', ''),
        gws.StyleValues(vals),
    )


def from_config(cfg: gws.Config, opts: parser.Options = None) -> 'Object':
    """Creates a css style object from a configuration.

        Args:
            cfg: A configuration.
            opts: Configuration options.

        Returns:
            A css style object.

        """
    return from_dict(
        gws.u.to_dict(cfg),
        opts or parser.Options(trusted=True, strict=True))


def from_props(props: gws.Props, opts: parser.Options = None) -> 'Object':
    """Creates a css style object from properties.

        Args:
            props: Properties.
            opts: Property Options.

        Returns:
            A css style object.

        """
    return from_dict(
        gws.u.to_dict(props),
        opts or parser.Options(trusted=False, strict=False))


##


class Config(gws.Config):
    """Feature style"""

    cssSelector: t.Optional[str]
    """CSS selector"""
    text: t.Optional[str]
    """raw style content"""
    values: t.Optional[dict]
    """style values"""


class Props(gws.Props):

    cssSelector: t.Optional[str]
    """CSS selector"""
    values: t.Optional[dict]
    """Style values"""


class Object(gws.Style):
    def __init__(self, selector, text, values):
        self.cssSelector = selector
        self.text = text
        self.values = values

    def props(self, user):
        ico = self.values.icon
        if ico and isinstance(ico, icon.ParsedIcon):
            ico = icon.to_data_url(ico)
        else:
            # NB if icon is not parsed, don't give it back
            ico = ''

        return Props(
            cssSelector=self.cssSelector or '',
            values=gws.u.merge(self.values, icon=ico),
        )
