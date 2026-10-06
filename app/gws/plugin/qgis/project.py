"""Loading and storing QGIS projects."""

import os

import gws
import gws.base.database
import gws.config.util
import gws.lib.jsonx
import gws.lib.datetimex
import gws.lib.xmlx
import gws.lib.zipx
import gws.lib.sa as sa

from . import caps


class Error(gws.Error):
    """QGIS project error."""

    pass


class StoreType(gws.Enum):
    """Where a QGIS project is stored."""

    file = 'file'
    """Project file (``.qgs`` or ``.qgz``)."""
    postgres = 'postgres'
    """Postgres table ``qgis_projects``."""


class Store(gws.Data):
    """Location of a QGIS project."""

    type: StoreType
    """Store type."""
    path: gws.FilePath
    """Project file path, for file stores."""
    dbUid: str
    """Database provider UID, for Postgres stores."""
    schema: str
    """Schema of the projects table, for Postgres stores. Defaults to ``public``."""
    projectName: str
    """Project name."""


_PRJ_EXT = '.qgs'
_ZIP_EXT = '.qgz'
_PRJ_TABLE = 'qgis_projects'


def from_store(root: gws.Root, store: Store) -> 'Object':
    """Load a QGIS project from a store.

    Args:
        root: Root object, used to find the database provider.
        store: Project location.

    Returns:
        The project.

    Raises:
        ``Error``: If the project cannot be found or loaded.
    """
    if store.type == StoreType.file:
        return from_path(store.path)
    if store.type == StoreType.postgres:
        return _from_db(root, store)
    raise Error(f'qgis project cannot be loaded')


def from_path(path: str) -> 'Object':
    """Load a QGIS project from a ``.qgs`` or ``.qgz`` file.

    Args:
        path: Project file path.

    Returns:
        The project.

    Raises:
        ``Error``: If a ``.qgz`` archive contains no project or the QGIS version is not supported.
    """
    if path.endswith(_ZIP_EXT):
        return _from_zipped_bytes(gws.u.read_file_b(path))
    return from_string(gws.u.read_file(path))


def from_string(text: str) -> 'Object':
    """Create a QGIS project from XML text.

    Args:
        text: Project XML.

    Returns:
        The project.

    Raises:
        ``Error``: If the QGIS version is not supported.
    """
    return Object(text)


def _from_zipped_bytes(b: bytes) -> 'Object':
    d = gws.lib.zipx.unzip_bytes_to_dict(b)
    for k, v in d.items():
        if k.endswith(_PRJ_EXT):
            return from_string(v.decode('utf8'))
    raise Error(f'no qgis project')


def _from_db(root: gws.Root, store: Store):
    db = root.app.databaseMgr.find_provider(ext_type='postgres', uid=store.dbUid)
    schema = store.get('schema') or 'public'
    tab = db.table(f'{schema}.{_PRJ_TABLE}')

    with db.begin() as conn:
        for row in conn.execute(sa.select(tab.c.content).where(tab.c.name == store.projectName)):
            return _from_zipped_bytes(row[0])
        raise Error(f'{store.projectName!r} not found')


def _to_db(root: gws.Root, store: Store, content: bytes):
    db = root.app.databaseMgr.find_provider(ext_type='postgres', uid=store.dbUid)
    schema = store.get('schema') or 'public'
    tab = db.table(f'{schema}.{_PRJ_TABLE}')

    metadata = {
        'last_modified_time': gws.lib.datetimex.to_iso_string(),
        'last_modified_user': 'GWS',
    }

    with db.begin() as conn:
        conn.execute(tab.delete().where(tab.c.name == store.projectName + '.bak'))
        conn.execute(tab.update().values(name=store.projectName + '.bak').where(tab.c.name == store.projectName))
        conn.execute(tab.insert().values(
            name=store.projectName,
            metadata=metadata,
            content=content,
        ))


class Object:
    """A QGIS project, held as XML text.

    Only QGIS 3 projects are supported. The parsed XML tree is created on
    demand and is not pickled.
    """

    version: str
    """QGIS version that wrote the project, e.g. ``3.34.0``."""
    sourceHash: str
    """SHA-256 hash of the project XML text."""

    def __init__(self, text: str):
        """Create a project from XML text.

        Args:
            text: Project XML.

        Raises:
            ``Error``: If the QGIS version is not supported.
        """
        self.text = text
        self.sourceHash = gws.u.sha256(self.text)

        ver = self.xml_root().get('version', '').split('-')[0]
        if not ver.startswith('3'):
            raise Error(f'unsupported qgis version {ver!r}')
        self.version = ver

    def __getstate__(self):
        """Return the pickle state without the parsed XML tree."""
        return gws.u.omit(vars(self), '_xml_root')

    def xml_root(self) -> gws.XmlElement:
        """Return the root element of the project XML.

        The XML is parsed on the first call. Changes to the tree are kept
        and written by ``to_xml``, ``to_path`` and ``to_store``.

        Returns:
            Root element.
        """
        if not hasattr(self, '_xml_root'):
            setattr(self, '_xml_root', gws.lib.xmlx.from_string(self.text))
        return getattr(self, '_xml_root')

    def to_store(self, root: gws.Root, store: Store):
        """Save the project to a store.

        In a Postgres store, an existing project with the same name is
        renamed to ``<name>.bak``, replacing an older backup.

        Args:
            root: Root object, used to find the database provider.
            store: Project location.

        Raises:
            ``Error``: If the store type is not supported.
        """
        if store.type == StoreType.file:
            return self.to_path(store.path)
        if store.type == StoreType.postgres:
            src = self.to_xml()
            name = store.projectName + _PRJ_EXT
            content = gws.lib.zipx.zip_to_bytes([{name: src}])
            return _to_db(root, store, content)
        raise Error(f'qgis project cannot be stored')

    def to_path(self, path: str):
        """Save the project to a file.

        If the path ends with ``.qgz``, the project is saved as a zip archive.

        Args:
            path: File path.
        """
        src = self.to_xml()
        if path.endswith(_ZIP_EXT):
            name = os.path.basename(path).replace(_ZIP_EXT, _PRJ_EXT)
            content = gws.lib.zipx.zip_to_bytes([{name: src}])
            gws.u.write_file_b(path, content)
        else:
            gws.u.write_file(path, src)

    def to_xml(self):
        """Serialize the project XML.

        Returns:
            Project XML text.
        """
        return self.xml_root().to_string()

    def caps(self) -> caps.Caps:
        """Parse the project capabilities.

        Returns:
            Project capabilities.

        Raises:
            ``gws.Error``: If the project CRS is invalid.
        """
        return caps.parse_element(self.xml_root())
