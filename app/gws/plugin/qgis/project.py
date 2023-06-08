"""Qgis Project API."""

import os

import gws
import gws.base.database
import gws.lib.jsonx
import gws.lib.date
import gws.lib.xmlx
import gws.lib.zipx

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


def from_storage(storage: Storage, obj: gws.INode):
    if storage.type == StorageType.file:
        return from_path(storage.path)
    if storage.type == StorageType.postgres:
        return _db_op('read', storage, obj)
    raise Error(f'qgis project cannot be loaded')


def from_path(path: str) -> 'Object':
    if path.endswith(_ZIP_EXT):
        return _from_bytes(gws.read_file_b(path))
    return from_string(gws.read_file(path))


def from_string(text: str) -> 'Object':
    return Object(text)


def to_storage(storage: Storage, text: str, obj: gws.INode):
    if storage.path:
        return to_path(storage.path, text)
    if storage.name:
        name = storage.name + _PRJ_EXT
        content = gws.lib.zipx.zip_to_bytes({name: text})
        return _db_op('write', storage, obj, content)
    raise Error(f'qgis project cannot be stored')


def to_path(path: str, text: str):
    if path.endswith(_ZIP_EXT):
        name = os.path.basename(path).replace(_ZIP_EXT, _PRJ_EXT)
        content = gws.lib.zipx.zip_to_bytes({name: text})
        gws.write_file_b(path, content)
    else:
        gws.write_file(path, text)


def _from_bytes(b: bytes) -> 'Object':
    d = gws.lib.zipx.unzip_bytes_to_dict(b)
    for k, v in d.items():
        if k.endswith(_PRJ_EXT):
            return from_string(v.decode('utf8'))
    raise Error(f'no qgis project')


def _db_op(op: str, storage: Storage, obj: gws.INode, content: bytes = None):
    prov = gws.base.database.provider.get_for(obj, storage.dbUid, 'postgres')
    schema = storage.get('schema') or 'public'
    table_name = f'{schema}.{_PRJ_TABLE}'

    with prov.session() as sess:
        tab = sess.autoload(table_name)
        if tab is None:
            raise Error(f'table {table_name!r} does not exist')

        if op == 'read':
            rs = sess.execute(tab.select().where(tab.c.name == storage.name))
            for r in rs.mappings().all():
                return _from_bytes(r['content'])
            raise Error(f'{storage.name!r} not found')

        if op == 'write':
            metadata = {
                'last_modified_time': gws.lib.date.now_iso(),
                'last_modified_user': 'GWS',
            }
            sess.execute(tab.delete().where(tab.c.name == storage.name + '.bak'))
            sess.execute(tab.update().values(name=storage.name + '.bak').where(tab.c.name == storage.name))
            sess.execute(tab.insert().values(
                name=storage.name,
                metadata=metadata,
                content=content,
            ))
            sess.execute(tab.delete().where(tab.c.name == storage.name + '.bak'))
            sess.commit()


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

    def to_storage(self, storage: Storage, obj: gws.INode):
        return to_storage(storage, self.to_xml(), obj)

    def to_path(self, path):
        to_path(path, self.to_xml())

    def to_xml(self):
        return self.xml_root().to_string()

    def caps(self) -> caps.Caps:
        if not hasattr(self, '_caps'):
            setattr(self, '_caps', caps.parse_element(self.xml_root()))
        return getattr(self, '_caps')
