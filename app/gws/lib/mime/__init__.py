"""MIME types.

Defines constants for the MIME types used in the application (``PNG``, ``JSON``, ``GML3``
and so on) and functions to normalize MIME types and to map them to and from file extensions.

All MIME types in the application go through ``get``. It accepts a MIME type in any spelling
(case, spaces, quoted parameter values), a content type with parameters, a known alias (e.g.
``application/vnd.ogc.gml`` or ``image/png8``) or a file extension, and returns the normalized
MIME type with its parameters: lower case, ``"; "`` between parameters, known types and aliases
mapped to the constants of this module (``application/gml+xml; version=3``). ``get_base`` returns
it without parameters. Lookups that are not covered by the built-in tables fall back to the
standard ``mimetypes`` module; unknown MIME types give an empty string.

Example::

    gws.lib.mime.get('image/jpg')  # 'image/jpeg'
    gws.lib.mime.get('png')  # 'image/png'
    gws.lib.mime.get('image/png8')  # 'image/png; mode=8bit'
    gws.lib.mime.get_base('image/png8')  # 'image/png'
    gws.lib.mime.for_path('/data/report.pdf')  # 'application/pdf'
    gws.lib.mime.extension_for(gws.lib.mime.JPEG)  # 'jpeg'
"""

import mimetypes

BIN = 'application/octet-stream'
CSS = 'text/css'
CSV = 'text/csv'
DOC = 'application/msword'
DOCX = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
FORM = 'application/x-www-form-urlencoded'
GEOJSON = 'application/geo+json'
GIF = 'image/gif'
GML = 'application/gml+xml'
GML2 = 'application/gml+xml; version=2'
GML3 = 'application/gml+xml; version=3'
GML32 = 'application/gml+xml; version=3.2'
GPKG = 'application/geopackage+sqlite3'
GZIP = 'application/gzip'
HTML = 'text/html'
JPEG = 'image/jpeg'
JPEG2000 = 'image/jp2'
JS = 'text/javascript'
JSON = 'application/json'
KML = 'application/vnd.google-earth.kml+xml'
KMZ = 'application/vnd.google-earth.kmz'
MSGPACK = 'application/msgpack'
MULTIPART = 'multipart/form-data'
MVT = 'application/vnd.mapbox-vector-tile'
PDF = 'application/pdf'
PNG = 'image/png'
PNG8 = 'image/png; mode=8bit'
PNG24 = 'image/png; mode=32bit'
PPT = 'application/vnd.ms-powerpoint'
PPTX = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
SVG = 'image/svg+xml'
TIFF = 'image/tiff'
TTF = 'font/ttf'
TXT = 'text/plain'
WEBP = 'image/webp'
WOFF2 = 'font/woff2'
XLS = 'application/vnd.ms-excel'
XLSX = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
XML = 'text/xml'
ZIP = 'application/zip'


_common = {globals()[k] for k in list(globals()) if k.isupper()}

_common_extensions = {
    'css': CSS,
    'csv': CSV,
    'doc': DOC,
    'docx': DOCX,
    'geojson': GEOJSON,
    'gif': GIF,
    'gml': GML,
    'gml2': GML2,
    'gpkg': GPKG,
    'html': HTML,
    'jpeg': JPEG,
    'jpg': JPEG,
    'js': JS,
    'json': JSON,
    'kml': KML,
    'kmz': KMZ,
    'mvt': MVT,
    'pdf': PDF,
    'png': PNG,
    'ppt': PPT,
    'pptx': PPTX,
    'svg': SVG,
    'tif': TIFF,
    'tiff': TIFF,
    'ttf': TTF,
    'txt': TXT,
    'webp': WEBP,
    'woff2': WOFF2,
    'xls': XLS,
    'xlsx': XLSX,
    'xml': XML,
    'zip': ZIP,
}

_aliases = {
    'application/gml:3': GML,
    'application/html': HTML,
    'application/javascript': JS,
    'application/json; subtype=geojson': GEOJSON,
    'application/vnd.geo+json': GEOJSON,
    'application/vnd.ogc.csw_xml': XML,
    'application/vnd.ogc.gml': GML2,
    'application/vnd.ogc.gml/3.1.1': GML3,
    'application/vnd.ogc.se_xml': XML,
    'application/vnd.ogc.wms_xml': XML,
    'application/x-font-ttf': TTF,
    'application/x-gzip': GZIP,
    'application/x-pdf': PDF,
    'application/x-protobuf': MVT,
    'application/xml': XML,
    'application/xml; subtype=gml/2': GML2,
    'application/xml; subtype=gml/3': GML,
    'application/xml; subtype=gml/3.1.1': GML3,
    'image/geotiff': TIFF,
    'image/jpg': JPEG,
    'image/png32': PNG24,
    'image/png8': PNG8,
    'image/svg': SVG,
    'text/xhtml': HTML,
    'text/xml; subtype=gml/3.1.1': GML3,
}


def get(mt: str) -> str:
    """Return the normalized MIME type, keeping its parameters.

    Known types and aliases are mapped to the constants of this module (``image/png8`` to ``PNG8``).
    Other parameters are kept: ``text/html;Charset=utf-8`` gives ``text/html; charset=utf-8``.

    Args:
        mt: MIME type in any spelling, content type, alias or file extension.

    Returns:
        The normalized MIME type, or an empty string if it is unknown.
    """

    if not mt:
        return ''

    s = _get_quick(mt)
    if s:
        return s

    mt = _normalize(mt)

    s = _get_quick(mt)
    if s:
        return s

    base, sep, params = mt.partition('; ')

    if sep:
        s = _get_quick(base)
        if s:
            return s if ';' in s else s + '; ' + params
        if mimetypes.guess_extension(base):
            return mt
        return ''

    if '/' in mt and mimetypes.guess_extension(mt):
        return mt

    t, _ = mimetypes.guess_type('x.' + mt)
    if t:
        return t

    return ''


def get_base(mt: str) -> str:
    """Return the normalized MIME type without parameters.

    Parameters are removed before and after the lookup, so that unknown parameters
    do not make a known type unknown.

    Example::

        get_base('image/png8')  # 'image/png'
        get_base('application/vnd.ogc.gml')  # 'application/gml+xml'

    Args:
        mt: MIME type in any spelling, content type, alias or file extension.

    Returns:
        The normalized MIME type without parameters, or an empty string if it is unknown.
    """

    b = mt.split(';')[0]
    b = get(b.strip())
    return b.split(';')[0]


def _normalize(mt: str) -> str:
    """Spell a MIME type canonically (RFC 9110, 8.3.1): lower-case names, no quotes, ``"; "`` between parameters."""

    parts = mt.split(';')
    res = [parts[0].replace(' ', '').lower()]
    for p in parts[1:]:
        name, _, value = p.partition('=')
        name = name.strip().lower()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        res.append(f'{name}={value}')
    return '; '.join(res)


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
