"""Feature styles.

A style describes how features are drawn: fill, stroke, markers, icons and labels.
Styles are written in a subset of CSS, extended with custom properties for markers and labels
(e.g. ``--marker``, ``--label-fill``). A leading ``--`` is optional, and dashes in property names
are converted to underscores, so ``--label-fill`` becomes the ``label_fill`` value in ``gws.StyleValues``.

Submodules:

- ``core``: the ``Object`` class implementing ``gws.Style``, its ``Config`` and ``Props``,
  and the constructors ``from_dict``, ``from_config`` and ``from_props``.
- ``parser``: parses CSS text or a dict of values into a dict of style values,
  validating property names and values and filling in defaults.
- ``icon``: loads and parses SVG icons from data URLs, files or URLs.

Parsing depends on whether the source is trusted. Styles from the configuration are trusted
and may load icons from arbitrary files and URLs. Styles from client requests are untrusted:
icons are only accepted as data URLs or from the configured image directories.
In strict mode, invalid properties and values raise an error, otherwise they are logged and skipped.

In style props sent to the client, a parsed icon is returned as a data URL.
An icon that has not been parsed is not returned.

Example config::

    style {
        text "fill: rgba(255,0,0,0.4); stroke: red; stroke-width: 2; --label-fill: black"
    }

Example::

    import gws.lib.style

    st = gws.lib.style.from_dict({'text': 'stroke: blue; stroke-width: 3px'})
    st.values.stroke_width  # 3
"""

from .core import (
    Config, Object, Props,
    from_dict,
    from_props,
    from_config
)
from . import icon, parser
