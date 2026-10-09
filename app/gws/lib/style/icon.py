"""SVG icon parser."""

from typing import Optional

import base64
import re
import urllib.parse

import gws
import gws.lib.net
import gws.lib.svg
import gws.lib.osx
import gws.lib.xmlx as xmlx


class Error(gws.Error):
    """Raised when an icon cannot be loaded or parsed."""

    pass


class ParsedIcon(gws.Data):
    """A parsed icon."""

    svg: gws.XmlElement
    """The normalized SVG element."""


def to_data_url(icon: ParsedIcon) -> str:
    """Convert a parsed icon to a base64 data URL.

    Args:
        icon: Parsed icon.

    Returns:
        A ``data:image/svg+xml;base64,...`` URL, or an empty string if the icon has no SVG.
    """

    if icon.svg:
        xml = icon.svg.to_string()
        return 'data:image/svg+xml;base64,' + base64.standard_b64encode(xml.encode('utf8')).decode('utf8')
    return ''


def parse(val: str, opts) -> Optional[ParsedIcon]:
    """Load and parse an SVG icon.

    The value can be wrapped in CSS ``url(...)`` and quotes. It is resolved as follows:

    - a ``data:`` URL is decoded,
    - a path is looked up in the image directories (``opts.imageDirs``),
    - in the trusted mode only, an ``http(s)`` URL is fetched, or any other value is read as a file path.

    The SVG is normalized (unsafe elements and attributes are removed) and must have a width and a height.

    Args:
        val: Data URL, URL or path.
        opts: Parser options (``gws.lib.style.parser.Options``).

    Returns:
        The parsed icon, or ``None`` if the value is empty or does not contain an SVG.

    Raises:
        ``Error``: If the value is untrusted, cannot be loaded or decoded, or the SVG is invalid.
    """
    if not val:
        return

    val = str(val).strip()
    m = re.match(r'^url\((.+?)\)$', val)
    if m:
        val = m.group(1)

    val = val.strip('\'\"')

    bs = _get_bytes(val, opts)
    if not bs:
        return

    if bs.startswith(b'<'):
        svg = _parse_svg(bs.decode('utf8'))
        if svg:
            return ParsedIcon(svg=svg)

    # @TODO other icon formats?


##


def _get_bytes(val, opts) -> Optional[bytes]:
    if val.startswith('data:'):
        return _decode_data_url(val)

    # if not trusted, looks in provided public dirs

    for img_dir in opts.get('imageDirs', []):
        path = gws.lib.osx.abs_web_path(val, img_dir)
        if path:
            return gws.u.read_file_b(path)

    # network and aribtrary files only in the trusted mode

    if not opts.get('trusted'):
        raise Error('untrusted value', val)

    if re.match(r'^https?:', val):
        try:
            return gws.lib.net.http_request(val).content
        except Exception as exc:
            raise Error('network error', val) from exc

    try:
        return gws.u.read_file_b(val)
    except Exception as exc:
        raise Error('file error', val) from exc


_PREFIXES = [
    'data:image/svg+xml;base64,',
    'data:image/svg+xml;utf8,',
    'data:image/svg;base64,',
    'data:image/svg;utf8,',
]


def _decode_data_url(val) -> Optional[bytes]:
    for pfx in _PREFIXES:
        if val.startswith(pfx):
            s = val[len(pfx):]
            try:
                if 'base64' in pfx:
                    return base64.b64decode(s, validate=True)
                else:
                    return urllib.parse.unquote(s).encode('utf8')
            except Exception as exc:
                raise Error('decode error', val) from exc


def _parse_svg(val):
    try:
        el = xmlx.from_string(val, gws.XmlOptions(removeNamespaces=True))
    except Exception as exc:
        raise Error('parse error', val) from exc

    el_clean = gws.lib.svg.normalize_element(el)

    w = el_clean.get('width')
    h = el_clean.get('height')

    if not w or not h:
        raise Error('missing width or height', val)

    return el_clean
