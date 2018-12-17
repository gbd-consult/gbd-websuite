"""Mime types."""

import mimetypes
import os

import gws

_common = {
    'json': 'application/json',
    'pdf': 'application/pdf',
    'xml': 'application/xml',
    'zip': 'application/zip',
    'gif': 'image/gif',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'svg': 'image/svg+xml',
    'css': 'text/css',
    'csv': 'text/csv',
    'html': 'text/html',
    'js': 'application/javascript',
    'ttf': 'application/x-font-ttf'
}

default_allowed = list(_common.values())


def get(key):
    if key in _common:
        return _common[key]
    t, _ = mimetypes.guess_type('x.' + key)
    return t


def for_path(path):
    gws.p(path)
    _, ext = os.path.splitext(path)
    if ext[1:] in _common:
        return _common[ext[1:]]
    t, _ = mimetypes.guess_type(path)
    gws.p(t)
    return t
