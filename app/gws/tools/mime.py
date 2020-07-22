"""Mime types."""

import mimetypes
import os


_common = {
    'css': 'text/css',
    'csv': 'text/csv',
    'gif': 'image/gif',
    'html': 'text/html',
    'jpeg': 'image/jpeg',
    'jpg': 'image/jpeg',
    'js': 'application/javascript',
    'json': 'application/json',
    'pdf': 'application/pdf',
    'png': 'image/png',
    'svg': 'image/svg+xml',
    'ttf': 'application/x-font-ttf',
    'txt': 'text/plain',
    'xml': 'application/xml',
    'zip': 'application/zip',

    'gml2': 'application/vnd.ogc.gml',
    'gml3': 'application/vnd.ogc.gml/3.1.1',
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
