from typing import Optional

import base64
import re

import gws
import gws.lib.net
import gws.lib.svg
import gws.lib.osx
import gws.lib.xmlx as xmlx


class Error(gws.Error):
    pass


class ParsedIcon(gws.Data):
    """Svg data."""
    svg: gws.XmlElement
    """Structure and attributes of the svg."""


def to_data_url(icon: ParsedIcon) -> str:
    """Converts a svg to url.

    Args:
        icon: Svg object to convert.

    Returns:
        The url, or an empty string if ``icon`` is not a svg.

    """

    if icon.svg:
        xml = icon.svg.to_string()
        return 'data:image/svg+xml;base64,' + base64.standard_b64encode(xml.encode('utf8')).decode('utf8')
    return ''


def parse(val: str, opts) -> Optional[ParsedIcon]:
    """Parses an url or directory path to a svg.

    Args:
        val: An url or directory path containing a svg.
        opts: Url or directory path options.

    Returns:
        The svg, if the url is trusted, or if the path is in a trusted directory.

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
        return _decode_data_url(val, opts)

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


def _decode_data_url(val, trusted) -> Optional[bytes]:
    for pfx in _PREFIXES:
        if val.startswith(pfx):
            s = val[len(pfx):]
            try:
                if 'base64' in pfx:
                    return base64.standard_b64decode(s)
                else:
                    return s.encode('utf8')
            except Exception as exc:
                raise Error('decode error', val) from exc


def _parse_svg(val):
    try:
        el = xmlx.from_string(val)
    except Exception as exc:
        raise Error('parse error', val) from exc

    el_clean = gws.lib.svg.sanitize_element(el)

    w = el_clean.get('width')
    h = el_clean.get('height')

    if not w or not h:
        raise Error('missing width or height', val)

    return el_clean
