"""Mime types."""

import mimetypes
import os

CSS = 'text/css'
CSV = 'text/csv'
GIF = 'image/gif'
HTML = 'text/html'
JPEG = 'image/jpeg'
JPG = 'image/jpeg'
JS = 'application/javascript'
JSON = 'application/json'
PDF = 'application/pdf'
PNG = 'image/png'
SVG = 'image/svg+xml'
TTF = 'application/x-font-ttf'
TXT = 'text/plain'
XML = 'application/xml'
ZIP = 'application/zip'
GML = 'application/vnd.ogc.gml'
GML3 = 'application/vnd.ogc.gml/3.1.1'

BIN = 'application/octet-stream'

_common = {
    'css': CSS,
    'csv': CSV,
    'gif': GIF,
    'html': HTML,
    'jpeg': JPEG,
    'jpg': JPG,
    'js': JS,
    'json': JSON,
    'pdf': PDF,
    'png': PNG,
    'svg': SVG,
    'ttf': TTF,
    'txt': TXT,
    'xml': XML,
    'zip': ZIP,
    'gml': GML,
    'gml3': GML3,
}

_aliases = {
    'text/xml': 'xml',
    'application/xml': 'xml',
}

DEFAULT_ALLOWED = list(_common.values())


def get(key):
    if not key:
        return

    key = key.lower()

    # literal mime type
    if '/' in key:
        if key in _aliases:
            return _common[_aliases[key]]
        return key

    # shortcut
    if key in _common:
        return _common[key]

    t, _ = mimetypes.guess_type('x.' + key)
    return t


def for_path(path):
    _, ext = os.path.splitext(path)
    if ext[1:] in _common:
        return _common[ext[1:]]
    t, _ = mimetypes.guess_type(path)
    return t


def extension(type):
    for k, v in _common.items():
        if v == type:
            return k
    t = mimetypes.guess_extension(type)
    if t:
        return t[1:]
