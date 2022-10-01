"""Qgis Project API."""

import zipfile

import gws
import gws.lib.xmlx

from . import caps


class Error(gws.Error):
    pass


# @TODO read from db
# @TODO read additional files (qlr, qml etc)

def from_path(path: str) -> 'Object':
    if not path.endswith('.qgz'):
        return from_string(gws.read_file(path), path)

    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.filename.endswith('.qgs'):
                with zf.open(info, 'r') as fp:
                    return from_string(fp.read().decode('utf8'), path)

    raise Error(f'cannot open qgis project {path!r}')


def from_string(xml: str, path: str = None) -> 'Object':
    return Object(xml, path or '')


class Object:
    rootElement: gws.IXmlElement
    path: str
    version: str
    sourceHash: str

    def __init__(self, xml: str, path: str):
        self.rootElement = gws.lib.xmlx.from_string(xml)
        self.path = path
        self.sourceHash = gws.sha256(xml)

        ver = self.rootElement.get('version', '').split('-')[0]
        if not ver.startswith('3'):
            raise Error(f'unsupported qgis version {ver!r}')
        self.version = ver

    def save(self, path=None):
        path = path or self.path
        if path.endswith('.qgz'):
            raise Error('writing qgz is not supported yet')
        gws.write_file(path, self.to_xml())

    def to_xml(self):
        return self.rootElement.to_string()

    def caps(self) -> caps.Caps:
        if not hasattr(self, '_caps'):
            setattr(self, '_caps', caps.parse_element(self.rootElement))
        return getattr(self, '_caps')
