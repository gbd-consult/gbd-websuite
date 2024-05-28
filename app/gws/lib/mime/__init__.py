"""Mime types."""

from typing import Optional

import mimetypes

BIN = 'application/octet-stream'
CSS = 'text/css'
CSV = 'text/csv'
GEOJSON = 'application/geojson'
GIF = 'image/gif'
GML = 'application/gml+xml'
GML2 = 'application/gml+xml;version=2'
GML3 = 'application/gml+xml;version=3'
GZIP = 'application/gzip'
HTML = 'text/html'
JPEG = 'image/jpeg'
JS = 'application/javascript'
JSON = 'application/json'
PDF = 'application/pdf'
PNG = 'image/png'
SVG = 'image/svg+xml'
TTF = 'application/x-font-ttf'
TXT = 'text/plain'
XML = 'application/xml'
ZIP = 'application/zip'
DOC = 'application/msword'
XLS = 'application/vnd.ms-excel'
PPT = 'application/vnd.ms-powerpoint'

_common = {
    BIN,
    CSS,
    CSV,
    GEOJSON,
    GIF,
    GML,
    GML2,
    GML3,
    GZIP,
    HTML,
    JPEG,
    JS,
    JSON,
    PDF,
    PNG,
    SVG,
    TTF,
    TXT,
    XML,
    ZIP,
    DOC,
    XLS,
    PPT,
}

_common_extensions = {
    'css': CSS,
    'csv': CSV,
    'gif': GIF,
    'gml': GML,
    'gml3': GML3,
    'html': HTML,
    'jpeg': JPEG,
    'jpg': JPEG,
    'js': JS,
    'json': JSON,
    'pdf': PDF,
    'png': PNG,
    'svg': SVG,
    'ttf': TTF,
    'txt': TXT,
    'xml': XML,
    'zip': ZIP,

    'doc': DOC,
    'xls': XLS,
    'ppt': PPT,
}

_aliases = {

    'application/vnd.ogc.gml': GML,
    'application/vnd.ogc.gml/3.1.1': GML3,
    'application/gml:3': GML3,
    'application/xml;subtype=gml/2': GML2,
    'application/xml;subtype=gml/3': GML3,

    'application/html': HTML,
    'application/x-gzip': GZIP,
    'application/x-pdf': PDF,
    'image/jpg': JPEG,
    'text/xhmtl': HTML,
    'text/xml': XML,

    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DOC,
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': XLS,
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': PPT,
}


def get(mt: str) -> Optional[str]:
    """Return the normalized mime type.

    Args:
        mt: Mime type or content type.

    Returns:
        The normalized mime type.
    """

    if not mt:
        return None

    mt = mt.strip().replace(' ', '').lower()

    s = _get_quick(mt)
    if s:
        return s

    for s, m in _aliases.items():
        if mt.startswith(s):
            return m

    if ';' in mt:
        p = mt.partition(';')
        s = _get_quick(p[0].strip())
        if s:
            return s

    if '/' in mt and mimetypes.guess_extension(mt):
        return mt

    t, _ = mimetypes.guess_type('x.' + mt)
    return t


def _get_quick(mt):
    if mt in _common:
        return mt
    if mt in _common_extensions:
        return _common_extensions[mt]
    if mt in _aliases:
        return _aliases[mt]


def for_path(path: str) -> str:
    """Returns the mime type for a given path.

    Args:
        path: Path to mime type.

    Returns:
        The mime type or ``BIN`` if type is unknown.
    """
    _, _, e = path.rpartition('.')
    if e in _common_extensions:
        return _common_extensions[e]
    t, _ = mimetypes.guess_type(path)
    return t or BIN


def extension_for(mt: str) -> Optional[str]:
    """Returns the extension of a given mime type.

    Args:
        mt: Mime type.

    Returns:
        The mime type extension.
    """

    for ext, rt in _common_extensions.items():
        if rt == mt:
            return ext
    s = mimetypes.guess_extension(mt)
    if s:
        return s[1:]
