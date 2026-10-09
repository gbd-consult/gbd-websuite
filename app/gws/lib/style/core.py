"""Style object and constructors."""

from typing import Optional

import gws

from . import parser, icon


# parsing depends on whenever the context is `trusted` (=config) or not (=request)


def from_dict(d: dict, opts: parser.Options = None) -> 'Object':
    """Create a style object from a dict.

    The dict can contain ``cssSelector``, ``text`` (CSS text) and ``values`` (a dict of style values).
    Values from ``values`` override those parsed from ``text``.

    Args:
        d: A dict with style properties.
        opts: Parser options. Defaults to untrusted and strict.

    Returns:
        A style object.

    Raises:
        ``gws.Error``: If a property or value is invalid and the parser is strict.
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
    """Create a style object from a configuration.

    Args:
        cfg: Style configuration.
        opts: Parser options. Defaults to trusted and strict.

    Returns:
        A style object.

    Raises:
        ``gws.Error``: If a property or value is invalid and the parser is strict.
    """
    return from_dict(
        gws.u.to_dict(cfg),
        opts or parser.Options(trusted=True, strict=True),
    )


def from_props(props: gws.Props, opts: parser.Options = None) -> 'Object':
    """Create a style object from client properties.

    Args:
        props: Style properties.
        opts: Parser options. Defaults to untrusted and non-strict.

    Returns:
        A style object.
    """
    return from_dict(
        gws.u.to_dict(props),
        opts or parser.Options(trusted=False, strict=False),
    )


##


class Config(gws.Config):
    """Feature style"""

    cssSelector: Optional[str]
    """CSS selector"""
    text: Optional[str]
    """raw style content"""
    values: Optional[dict]
    """style values"""


class Props(gws.Props):
    cssSelector: Optional[str]
    """CSS selector"""
    values: Optional[dict]
    """Style values"""


class Object(gws.Style):
    """Style object."""

    def __init__(self, selector, text, values):
        """Create a style object.

        Args:
            selector: CSS selector.
            text: Raw CSS text.
            values: Parsed style values.
        """
        self.cssSelector = selector
        self.text = text
        self.values = values

    def props(self, user):
        values = self.values
        if self.values.icon:
            values = gws.u.merge(self.values, icon=icon.to_data_url(self.values.icon))
        return Props(
            cssSelector=self.cssSelector or '',
            values=values,
        )
