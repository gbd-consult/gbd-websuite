"""Qgis Project API."""

import os

import gws
import gws.base.database
import gws.lib.jsonx
import gws.lib.date
import gws.lib.xmlx
import gws.lib.zipx
import gws.lib.sa as sa

from . import caps


class Error(gws.Error):
    pass


class StorageType(gws.Enum):
    file = 'file'
    postgres = 'postgres'


class Storage(gws.Data):
    type: StorageType
    path: gws.FilePath
    dbUid: str
    schema: str
    name: str


_PRJ_EXT = '.qgs'
_ZIP_EXT = '.qgz'
_PRJ_TABLE = 'qgis_projects'


def from_storage(root: gws.IRoot, storage: Storage) -> 'Object':
    if storage.type == StorageType.file:
        return from_path(storage.path)
    if storage.type == StorageType.postgres:
        return _db_read(root, storage)
    raise Error(f'qgis project cannot be loaded')


def from_path(path: str) -> 'Object':
    if path.endswith(_ZIP_EXT):
        return _from_bytes(gws.read_file_b(path))
    return from_string(gws.read_file(path))


def from_string(text: str) -> 'Object':
    return Object(text)


def _from_bytes(b: bytes) -> 'Object':
    d = gws.lib.zipx.unzip_bytes_to_dict(b)
    for k, v in d.items():
        if k.endswith(_PRJ_EXT):
            return from_string(v.decode('utf8'))
    raise Error(f'no qgis project')


def _db_read(root: gws.IRoot, storage: Storage):
    prov = gws.base.database.provider.get_for(root.app, storage.dbUid, 'postgres')
    schema = storage.get('schema') or 'public'
    tab = prov.table(f'{schema}.{_PRJ_TABLE}')

    with prov.connection() as conn:
        for row in conn.execute(sa.select(tab.c.content).where(tab.c.name.__eq__(storage.name))):
            return _from_bytes(row[0])
        raise Error(f'{storage.name!r} not found')


def _db_write(root: gws.IRoot, storage: Storage, content: bytes):
    prov = gws.base.database.provider.get_for(root.app, storage.dbUid, 'postgres')
    schema = storage.get('schema') or 'public'
    tab = prov.table(f'{schema}.{_PRJ_TABLE}')

    metadata = {
        'last_modified_time': gws.lib.date.now_iso(),
        'last_modified_user': 'GWS',
    }

    with prov.connection() as conn:
        conn.execute(tab.delete().where(tab.c.name.__eq__(storage.name + '.bak')))
        conn.execute(tab.update().values(name=storage.name + '.bak').where(tab.c.name.__eq__(storage.name)))
        conn.execute(tab.insert().values(
            name=storage.name,
            metadata=metadata,
            content=content,
        ))
        conn.commit()


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
        return gws.omit(vars(self), '_xml_root')

    def xml_root(self) -> gws.IXmlElement:
        if not hasattr(self, '_xml_root'):
            setattr(self, '_xml_root', gws.lib.xmlx.from_string(self.text))
        return getattr(self, '_xml_root')

    def to_storage(self, root: gws.IRoot, storage: Storage):
        if storage.path:
            return self.to_path(storage.path)
        if storage.name:
            src = self.to_xml()
            name = storage.name + _PRJ_EXT
            content = gws.lib.zipx.zip_to_bytes({name: src})
            return _db_write(root, storage, content)
        raise Error(f'qgis project cannot be stored')

    def to_path(self, path: str):
        src = self.to_xml()
        if path.endswith(_ZIP_EXT):
            name = os.path.basename(path).replace(_ZIP_EXT, _PRJ_EXT)
            content = gws.lib.zipx.zip_to_bytes({name: src})
            gws.write_file_b(path, content)
        else:
            gws.write_file(path, src)

    def to_xml(self):
        return self.xml_root().to_string()

    def caps(self) -> caps.Caps:
        if not hasattr(self, '_caps'):
            setattr(self, '_caps', caps.parse_element(self.xml_root()))
        return getattr(self, '_caps')
