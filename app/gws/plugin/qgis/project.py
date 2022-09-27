"""Qgis Project API."""

import zipfile

import gws
import gws.lib.xmlx
import gws.types as t

from . import caps


class Error(gws.Error):
    pass


def from_path(path: str) -> 'Project':
    if not path.endswith('.qgz'):
        return from_string(gws.read_file(path), path)

    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.filename.endswith('.qgs'):
                with zf.open(info, 'r') as fp:
                    return from_string(fp.read().decode('utf8'), path)

    raise Error(f'cannot open qgis project {path!r}')


def from_string(xml: str, path: str = None) -> 'Project':
    return Project(xml, path or '')


class Project:
    rootElement: gws.IXmlElement
    path: str
    version: str

    def __init__(self, xml: str, path: str):
        self.rootElement = gws.lib.xmlx.from_string(xml)
        self.path = path

        ver = self.rootElement.get('version', '').split('-')[0]
        if not ver.startswith('3'):
            raise Error(f'unsupported qgis version {ver!r}')
        self.version = ver
        self._caps = None

    def save(self, path=None):
        path = path or self.path
        if path.endswith('.qgz'):
            raise Error('writing qgz is not supported yet')
        gws.write_file(path, self.to_xml())

    def to_xml(self):
        return self.rootElement.to_string()

    def caps(self) -> caps.Caps:
        if not self._caps:
            self._caps = caps.parse_element(self.rootElement)
        return self._caps
