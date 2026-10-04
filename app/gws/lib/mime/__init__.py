"""MIME types.

Defines constants for the MIME types used in the application (``PNG``, ``JSON``, ``GML3``
and so on) and functions to normalize MIME types and to map them to and from file extensions.

``get`` accepts a MIME type, a content type with parameters, a known alias (e.g.
``application/vnd.ogc.gml`` or ``image/jpg``) or a file extension, and returns a normalized
MIME type. Lookups that are not covered by the built-in tables fall back to the standard
``mimetypes`` module.

Example::

    gws.lib.mime.get('image/jpg')                   # 'image/jpeg'
    gws.lib.mime.get('png')                         # 'image/png'
    gws.lib.mime.for_path('/data/report.pdf')       # 'application/pdf'
    gws.lib.mime.extension_for(gws.lib.mime.JPEG)   # 'jpeg'
"""

from typing import Optional

import mimetypes

BIN = 'application/octet-stream'
CSS = 'text/css'
CSV = 'text/csv'
DOC = 'application/msword'
GEOJSON = 'application/geo+json'
GIF = 'image/gif'
GML = 'application/gml+xml'
GML2 = 'application/gml+xml;version=2'
GML3 = 'application/gml+xml;version=3'
GZIP = 'application/gzip'
HTML = 'text/html'
JPEG = 'image/jpeg'
JS = 'application/javascript'
JSON = 'application/json'
KML = 'application/vnd.google-earth.kml+xml'
PDF = 'application/pdf'
PNG = 'image/png'
PPT = 'application/vnd.ms-powerpoint'
SVG = 'image/svg+xml'
TTF = 'application/x-font-ttf'
TXT = 'text/plain'
WEBP = 'image/webp'
XLS = 'application/vnd.ms-excel'
XML = 'text/xml'
ZIP = 'application/zip'


_common = {
    BIN,
    CSS,
    CSV,
    DOC,
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
    KML,
    PDF,
    PNG,
    PPT,
    SVG,
    TTF,
    TXT,
    XLS,
    XML,
    ZIP,
}

_common_extensions = {
    'css': CSS,
    'csv': CSV,
    'doc': DOC,
    'gif': GIF,
    'gml': GML,
    'gml2': GML2,
    'html': HTML,
    'jpeg': JPEG,
    'jpg': JPEG,
    'js': JS,
    'json': JSON,
    'kml': KML,
    'pdf': PDF,
    'png': PNG,
    'ppt': PPT,
    'svg': SVG,
    'ttf': TTF,
    'txt': TXT,
    'xls': XLS,
    'xml': XML,
    'zip': ZIP,
}

_aliases = {
    'application/vnd.ogc.gml': GML2,
    'application/vnd.ogc.gml/3.1.1': GML,
    'application/gml:3': GML,
    'application/xml;subtype=gml/2': GML2,
    'application/xml;subtype=gml/3': GML,
    'application/html': HTML,
    'application/x-gzip': GZIP,
    'application/x-pdf': PDF,
    'image/jpg': JPEG,
    'text/xhtml': HTML,
    'application/xml': XML,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DOC,
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': XLS,
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': PPT,
}


def get(mt: str) -> Optional[str]:
    """Return the normalized MIME type.

    Args:
        mt: MIME type, content type, alias or file extension. Case and spaces are ignored.

    Returns:
        The normalized MIME type, or ``None`` if it is unknown.
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
    """Look up a MIME type, an extension or an alias in the built-in tables."""
    if mt in _common:
        return mt
    if mt in _common_extensions:
        return _common_extensions[mt]
    if mt in _aliases:
        return _aliases[mt]


def for_path(path: str) -> str:
    """Return the MIME type for a file path, based on its extension.

    Args:
        path: File path or name.

    Returns:
        The MIME type, or ``BIN`` if it is unknown.
    """
    _, _, e = path.rpartition('.')
    if e in _common_extensions:
        return _common_extensions[e]
    t, _ = mimetypes.guess_type(path)
    return t or BIN


def extension_for(mt: str, default: str = 'bin') -> str:
    """Return the file extension for a MIME type.

    Args:
        mt: Normalized MIME type.
        default: Extension to return if the MIME type is unknown.

    Returns:
        The extension, without a dot.
    """

    for ext, rt in _common_extensions.items():
        if rt == mt:
            return ext
    s = mimetypes.guess_extension(mt)
    if s:
        return s[1:]
    return default
