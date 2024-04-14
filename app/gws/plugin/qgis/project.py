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


class StoreType(gws.Enum):
    file = 'file'
    postgres = 'postgres'


class Store(gws.Data):
    type: StoreType
    path: gws.FilePath
    dbUid: str
    schema: str
    projectName: str


_PRJ_EXT = '.qgs'
_ZIP_EXT = '.qgz'
_PRJ_TABLE = 'qgis_projects'


def from_store(root: gws.Root, store: Store) -> 'Object':
    if store.type == StoreType.file:
        return from_path(store.path)
    if store.type == StoreType.postgres:
        return _from_db(root, store)
    raise Error(f'qgis project cannot be loaded')


def from_path(path: str) -> 'Object':
    if path.endswith(_ZIP_EXT):
        return _from_zipped_bytes(gws.u.read_file_b(path))
    return from_string(gws.u.read_file(path))


def from_string(text: str) -> 'Object':
    return Object(text)


def _from_zipped_bytes(b: bytes) -> 'Object':
    d = gws.lib.zipx.unzip_bytes_to_dict(b)
    for k, v in d.items():
        if k.endswith(_PRJ_EXT):
            return from_string(v.decode('utf8'))
    raise Error(f'no qgis project')


def _from_db(root: gws.Root, store: Store):
    prov = gws.base.database.provider.get_for(root.app, store.dbUid, 'postgres')
    schema = store.get('schema') or 'public'
    tab = prov.table(f'{schema}.{_PRJ_TABLE}')

    with prov.connection() as conn:
        for row in conn.execute(sa.select(tab.c.content).where(tab.c.name.__eq__(store.projectName))):
            return _from_zipped_bytes(row[0])
        raise Error(f'{store.projectName!r} not found')


def _to_db(root: gws.Root, store: Store, content: bytes):
    prov = gws.base.database.provider.get_for(root.app, store.dbUid, 'postgres')
    schema = store.get('schema') or 'public'
    tab = prov.table(f'{schema}.{_PRJ_TABLE}')

    metadata = {
        'last_modified_time': gws.lib.date.now_iso(),
        'last_modified_user': 'GWS',
    }

    with prov.connection() as conn:
        conn.execute(tab.delete().where(tab.c.name.__eq__(store.projectName + '.bak')))
        conn.execute(tab.update().values(name=store.projectName + '.bak').where(tab.c.name.__eq__(store.projectName)))
        conn.execute(tab.insert().values(
            name=store.projectName,
            metadata=metadata,
            content=content,
        ))
        conn.commit()


class Object:
    version: str
    sourceHash: str

    def __init__(self, text: str):
        self.text = text
        self.sourceHash = gws.u.sha256(self.text)

        ver = self.xml_root().get('version', '').split('-')[0]
        if not ver.startswith('3'):
            raise Error(f'unsupported qgis version {ver!r}')
        self.version = ver

    def __getstate__(self):
        return gws.u.omit(vars(self), '_xml_root')

    def xml_root(self) -> gws.XmlElement:
        if not hasattr(self, '_xml_root'):
            setattr(self, '_xml_root', gws.lib.xmlx.from_string(self.text))
        return getattr(self, '_xml_root')

    def to_store(self, root: gws.Root, store: Store):
        if store.path:
            return self.to_path(store.path)
        if store.projectName:
            src = self.to_xml()
            name = store.projectName + _PRJ_EXT
            content = gws.lib.zipx.zip_to_bytes({name: src})
            return _to_db(root, store, content)
        raise Error(f'qgis project cannot be stored')

    def to_path(self, path: str):
        src = self.to_xml()
        if path.endswith(_ZIP_EXT):
            name = os.path.basename(path).replace(_ZIP_EXT, _PRJ_EXT)
            content = gws.lib.zipx.zip_to_bytes({name: src})
            gws.u.write_file_b(path, content)
        else:
            gws.u.write_file(path, src)

    def to_xml(self):
        return self.xml_root().to_string()

    def caps(self) -> caps.Caps:
        return caps.parse_element(self.xml_root())
