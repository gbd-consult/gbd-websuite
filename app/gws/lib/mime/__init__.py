"""Mime types."""

import mimetypes

import gws.types as t

BIN = 'application/octet-stream'
"""MIME-Type for files ending in .bin"""
CSS = 'text/css'
"""MIME-Type for files ending in .CSS"""
CSV = 'text/csv'
"""MIME-Type for files ending in .CSV"""
GEOJSON = 'application/geojson'
"""MIME-Type for files ending in .GEOJSON"""
GIF = 'image/gif'
"""MIME-Type for files ending in .GIF"""
GML = 'application/gml+xml'
"""MIME-Type for files ending in .GML"""
GML2 = 'application/gml+xml; version=2'
"""MIME-Type for files ending in .GML2"""
GML3 = 'application/gml+xml; version=3'
"""MIME-Type for files ending in .GML3"""
GZIP = 'application/gzip'
"""MIME-Type for files ending in .GZIP"""
HTML = 'text/html'
"""MIME-Type for files ending in .HTML"""
JPEG = 'image/jpeg'
"""MIME-Type for files ending in .JPEG"""
JS = 'application/javascript'
"""MIME-Type for files ending in .JS"""
JSON = 'application/json'
"""MIME-Type for files ending in .JSON"""
PDF = 'application/pdf'
"""MIME-Type for files ending in .PDF"""
PNG = 'image/png'
"""MIME-Type for files ending in .PNG"""
SVG = 'image/svg+xml'
"""MIME-Type for files ending in .SVG"""
TTF = 'application/x-font-ttf'
"""MIME-Type for files ending in .TTF"""
TXT = 'text/plain'
"""MIME-Type for files ending in .TXT"""
XML = 'application/xml'
"""MIME-Type for files ending in .XML"""
ZIP = 'application/zip'
"""MIME-Type for files ending in .ZIP"""

_common = {
    BIN,
    CSS,
    CSV,
    GEOJSON,
    GIF,
    GML,
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

    'doc': 'application/msword',
    'xls': 'application/vnd.ms-excel',
    'ppt': 'application/vnd.ms-powerpoint',
}

_aliases = {

    'application/vnd.ogc.gml': GML,
    'application/vnd.ogc.gml/3.1.1': GML3,
    'application/gml:3': GML3,
    'application/xml; subtype=gml/2': GML2,
    'application/xml; subtype=gml/3': GML3,

    'application/html': HTML,
    'application/x-gzip': GZIP,
    'application/x-pdf': PDF,
    'image/jpg': JPEG,
    'text/xhmtl': HTML,
    'text/xml': XML,

    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'application/msword',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'application/vnd.ms-powerpoint',
}


def get(mt: str) -> t.Optional[str]:
    """Return the normalized mime type.

    Args:
        mt: Mime type or content type.

    Returns:
        The normalized mime type.
    """

    if not mt:
        return None

    mt = mt.lower()

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


def extension_for(mt: str) -> t.Optional[str]:
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
