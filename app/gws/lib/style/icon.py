import base64
import re

import gws
import gws.lib.net
import gws.lib.svg
import gws.lib.xml2 as xml2
import gws.types as t


class Error(gws.Error):
    pass


class ParsedIcon(gws.Data):
    svg: gws.XmlElement


def to_data_url(icon: ParsedIcon) -> str:
    if icon.svg:
        xml = gws.lib.xml2.to_string(icon.svg)
        return 'data:image/svg+xml;base64,' + base64.standard_b64encode(xml.encode('utf8')).decode('utf8')
    return ''


def parse(val, trusted):
    if not val:
        return

    val = str(val).strip()
    m = re.match(r'^url\((.+?)\)$', val)
    if m:
        val = m.group(1)

    val = val.strip('\'\"')

    bs = _get_bytes(val, trusted)
    if not bs:
        return

    if bs.startswith(b'<'):
        svg = _parse_svg(bs.decode('utf8'))
        if svg:
            return ParsedIcon(svg=svg)

    # @TODO other icon formats?


##


def _get_bytes(val, trusted) -> t.Optional[bytes]:
    if val.startswith('data:'):
        return _decode_data_url(val, trusted)

    if not trusted:
        raise Error('untrusted value', val)

    if re.match(r'^https?:', val):
        try:
            return gws.lib.net.http_request(val).content
        except Exception as exc:
            raise Error('network error', val) from exc

    try:
        return gws.read_file_b(val)
    except Exception as exc:
        raise Error('file error', val) from exc


_PREFIXES = [
    'data:image/svg+xml;base64,',
    'data:image/svg+xml;utf8,',
    'data:image/svg;base64,',
    'data:image/svg;utf8,',
]


def _decode_data_url(val, trusted) -> t.Optional[bytes]:
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
        el = xml2.from_string(val)
    except Exception as exc:
        raise Error('parse error', val) from exc

    el_clean = gws.lib.svg.sanitize_element(el)

    w = xml2.attr(el_clean, 'width')
    h = xml2.attr(el_clean, 'height')

    if not w or not h:
        raise Error('missing width or height', val)

    return el_clean
