"""Backend for file system operations."""

import re
import os

import gws
import gws.common.action
import gws.tools.os2
import gws.web.error
import gws.tools.json2
import gws.tools.date
import gws.tools.sqlite

import gws.types as t


class Config(t.WithTypeAndAccess):
    """File system action"""

    root: t.DirPath  #: file system root


class WriteParams(t.Params):
    path: str
    data: bytes


class WriteResponse(t.Response):
    pass


class ReadParams(t.Params):
    path: str


class ReadResponse(t.Response):
    data: bytes


class ListParams(t.Params):
    pass


class ListEntry(t.Data):
    path: str


class ListResponse(t.Response):
    entries: t.List[ListEntry]


class DeleteParams(t.Params):
    path: str


class DeleteResponse(t.Response):
    pass


class UndeleteParams(t.Params):
    path: str


class UndeleteResponse(t.Response):
    pass


class EmptyTrashParams(t.Params):
    pass


class EmptyTrashResponse(t.Response):
    pass


TRASH_NAME = '__fs_trash'
DB_NAME = '__fs_meta6.sqlite'


class Object(gws.common.action.Object):
    def configure(self):
        super().configure()
        self.root_dir = gws.ensure_dir(self.var('root'))
        self.trash_dir = gws.ensure_dir(self.root_dir + '/' + TRASH_NAME)
        self.db_path = self.root_dir + '/' + DB_NAME

        with self._connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS meta(
                path TEXT,
                created_by TEXT,
                created_time DATETIME,
                updated_by TEXT,
                updated_time DATETIME,
                deleted INTEGER,
                PRIMARY KEY (path)
            ) WITHOUT ROWID''')

        os.chown(self.db_path, gws.UID, gws.GID)

    def api_write(self, req: t.IRequest, p: WriteParams) -> WriteResponse:
        """Write data to a new or existing file."""

        dp, fname = self._check_file_path(p.path)
        path = dp + '/' + fname
        meta = self._read_metadata(path)

        gws.ensure_dir(dp)

        if not meta:
            meta = {
                'created_by': req.user.fid,
                'created_time': gws.tools.date.now(),
            }

        if meta.get('deleted'):
            self._unlink(path)

        meta['updated_by'] = req.user.fid
        meta['updated_time'] = gws.tools.date.now()
        meta['deleted'] = False

        gws.write_file_b(path, p.data)
        self._write_metadata(path, meta)

        return WriteResponse()

    def api_read(self, req: t.IRequest, p: ReadParams) -> ReadResponse:
        """Read from an existing file."""

        dp, fname = self._check_file_path(p.path)
        path = dp + '/' + fname

        if not gws.tools.os2.is_file(path):
            raise gws.web.error.NotFound()

        meta = self._read_metadata(path) or self._metadata_from_path(path)
        # @TODO check permissions

        return ReadResponse(data=gws.read_file_b(path))

    def api_delete(self, req: t.IRequest, p: DeleteParams) -> DeleteResponse:
        """Move a file to trash."""

        dp, fname = self._check_file_path(p.path)
        path = dp + '/' + fname

        if not gws.tools.os2.is_file(path):
            raise gws.web.error.NotFound()

        meta = self._read_metadata(path) or self._metadata_from_path(path)
        # @TODO check permissions

        os.rename(path, self._trash_path(path))
        meta['deleted'] = True
        self._write_metadata(path, meta)

        return DeleteResponse()

    def api_list(self, req: t.IRequest, p: ListParams) -> ListResponse:
        """Return a list of all server files."""

        entries = []

        for p in gws.tools.os2.find_files(self.root_dir):
            p = gws.tools.os2.rel_path(p, self.root_dir)
            if p.startswith('__'):
                continue
            entries.append(ListEntry(path=p))

        return ListResponse(entries=entries)

    def api_undelete(self, req: t.IRequest, p: UndeleteParams) -> UndeleteResponse:
        """Restore a file from the trash."""

        dp, fname = self._check_file_path(p.path)
        path = dp + '/' + fname
        meta = self._read_metadata(path)

        if not meta or not meta['deleted']:
            raise gws.web.error.NotFound()

        os.rename(self._trash_path(path), meta['path'])
        meta['deleted'] = False
        self._write_metadata(path, meta)

        return UndeleteResponse()

    def api_list_trash(self, req: t.IRequest, p: ListParams) -> ListResponse:
        """List paths currently in the trash."""

        entries = []

        with self._connect() as conn:
            rs = conn.execute('SELECT * FROM meta WHERE deleted=1')
            for r in rs:
                entries.append(ListEntry(path=r['path']))

        return ListResponse(entries=entries)

    def api_empty_trash(self, req: t.IRequest, p: EmptyTrashParams) -> EmptyTrashResponse:
        """Empty the trash."""

        with self._connect() as conn:
            conn.execute('DELETE * FROM meta WHERE deleted=1')
        for p in gws.tools.os2.find_files(self.trash_dir):
            gws.tools.os2.unlink(p)

        return EmptyTrashResponse()

    ##

    def _check_file_path(self, path):
        pp = gws.tools.os2.parse_path(path)
        dp = gws.tools.os2.abs_path(pp['dirname'], self.root_dir)

        if not dp:
            raise gws.web.error.BadRequest('invalid path')

        if not re.match(r'^\w{1,60}', pp['name']):
            raise gws.web.error.BadRequest('invalid filename')

        if not re.match(r'^\w{1,60}', pp['extension']):
            raise gws.web.error.BadRequest('invalid filename')

        return dp, f"{pp['name']}.{pp['extension']}"

    def _read_metadata(self, path):
        with self._connect() as conn:
            rs = conn.execute('SELECT * FROM meta WHERE path=? LIMIT 1', [path])
            for r in rs:
                return dict(r)

    def _metadata_from_path(self, path):
        if not gws.tools.os2.is_file(path):
            return None
        return {
            'path': path,
            'created_by': 'sys::root',
            'created_time': gws.tools.date.now(),
            'updated_by': 'sys::root',
            'updated_time': gws.tools.date.now(),
            'deleted': False,
        }

    def _write_metadata(self, path, meta):
        meta['path'] = path
        keys = ','.join(meta)
        vals = ','.join(['?'] * len(meta))
        sql = f'INSERT OR REPLACE INTO meta ({keys}) VALUES ({vals})'

        with self._connect() as conn:
            conn.execute(sql, list(meta.values()))

    def _unlink(self, path):
        p = self._trash_path(path)
        gws.tools.os2.unlink(p)
        with self._connect() as conn:
            conn.execute('DELETE FROM meta WHERE path=?', [path])

    def _trash_path(self, path):
        return self.trash_dir + '/' + re.sub(r'\W', '__', path)

    def _connect(self):
        return gws.tools.sqlite.connect(self.db_path)
