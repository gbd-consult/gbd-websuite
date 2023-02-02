"""Qgis Project API."""

import zipfile
import sqlalchemy as sa

import gws
import gws.base.database
import gws.base.database.sql
import gws.lib.xmlx
import gws.lib.zipx
import gws.types as t

from . import caps


class Error(gws.Error):
    pass


class Source(gws.Data):
    path: t.Optional[gws.FilePath]
    db: t.Optional[str]
    schema: t.Optional[str]
    name: t.Optional[str]


def from_source(source: Source, obj: gws.INode):
    if source.path:
        return from_path(source.path)

    if source.name:
        prov = gws.base.database.provider.get_for(obj, source.db, 'postgres')
        schema = source.get('schema', 'public')
        table_name = f'{schema}.qgis_projects'
        with prov.session() as sess:
            tab = sess.autoload(table_name)
            if tab is None:
                raise Error(f'table {table_name!r} does not exist')
            rs = sess.execute(tab.select().where(tab.c.name == source.name))
            for r in rs.mappings().all():
                return from_bytes(r['content'])
            raise Error(f'{source.name!r} not found')

    raise Error(f'cannot load qgis project')


def from_bytes(b: bytes) -> 'Object':
    d = gws.lib.zipx.unzip_bytes_to_dict(b)
    for k, v in d.items():
        if k.endswith('.qgs'):
            return from_string(v.decode('utf8'))
    raise Error(f'no qgis project')


def from_path(path: str) -> 'Object':
    if path.endswith('.qgz'):
        return from_bytes(gws.read_file_b(path))
    return from_string(gws.read_file(path))


def from_string(text: str) -> 'Object':
    return Object(text)


class Object:
    version: str
    sourceHash: str

    def __init__(self, text: str):
        self.text = text
        self.sourceHash = gws.sha256(self.text)

        ver = self.xml_root().get('version', '').split('-')[0]
        if not ver.startswith('3'):
            raise Error(f'unsupported qgis version {ver!r}')
        self.version = ver

    def __getstate__(self):
        d = dict(vars(self))
        d.pop('_xml_root', None)
        return d

    def xml_root(self) -> gws.IXmlElement:
        if not hasattr(self, '_xml_root'):
            setattr(self, '_xml_root', gws.lib.xmlx.from_string(self.text))
        return getattr(self, '_xml_root')

    def to_path(self, path):
        if path.endswith('.qgz'):
            raise Error('writing qgz is not supported yet')
        gws.write_file(path, self.to_xml())

    def to_xml(self):
        return self.xml_root().to_string()

    def caps(self) -> caps.Caps:
        if not hasattr(self, '_caps'):
            setattr(self, '_caps', caps.parse_element(self.xml_root()))
        return getattr(self, '_caps')
