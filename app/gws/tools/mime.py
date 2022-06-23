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

    'gml': 'application/vnd.ogc.gml',
    'gml3': 'application/vnd.ogc.gml/3.1.1',

    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types

    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'xls': 'application/vnd.ms-excel',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'ppt': 'application/vnd.ms-powerpoint',
    'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',

    'odp': 'application/vnd.oasis.opendocument.presentation',
    'ods': 'application/vnd.oasis.opendocument.spreadsheet',
    'odt': 'application/vnd.oasis.opendocument.text',
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
