"""Backend for file system operations."""

import os
import re

import gws
import gws.types as t
import gws.base.api
import gws.lib.date
import gws.lib.json2
import gws.lib.os2
import gws.lib.sqlite
import gws.base.web.error


class Config(gws.WithAccess):
    """File system action"""

    root: gws.DirPath  #: file system root


class WriteParams(gws.Params):
    path: str
    data: bytes


class WriteResponse(gws.Response):
    pass


class ReadParams(gws.Params):
    path: str


class ReadResponse(gws.Response):
    data: bytes


class ListParams(gws.Params):
    pass


class ListEntry(gws.Data):
    path: str


class ListResponse(gws.Response):
    entries: list[ListEntry]


class DeleteParams(gws.Params):
    path: str


class DeleteResponse(gws.Response):
    pass


class UndeleteParams(gws.Params):
    path: str


class UndeleteResponse(gws.Response):
    pass


class EmptyTrashParams(gws.Params):
    pass


class EmptyTrashResponse(gws.Response):
    pass


TRASH_NAME = '__fs_trash'
DB_NAME = '__fs_meta6.sqlite'


class Object(gws.base.api.Action):
    def configure(self):
        
        self.root_dir = gws.ensure_dir(self.cfg('root'))
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

    def api_write(self, req: gws.IWebRequest, p: WriteParams) -> WriteResponse:
        """Write data to a new or existing file."""

        dp, fname = self._check_file_path(p.path)
        path = dp + '/' + fname
        meta = self._read_metadata(path)

        gws.ensure_dir(dp)

        if not meta:
            meta = {
                'created_by': req.user.fid,
                'created_time': gws.lib.date.now(),
            }

        if meta.get('deleted'):
            self._unlink(path)

        meta['updated_by'] = req.user.fid
        meta['updated_time'] = gws.lib.date.now()
        meta['deleted'] = False

        gws.write_file_b(path, p.data)
        self._write_metadata(path, meta)

        return WriteResponse()

    def api_read(self, req: gws.IWebRequest, p: ReadParams) -> ReadResponse:
        """Read from an existing file."""

        dp, fname = self._check_file_path(p.path)
        path = dp + '/' + fname

        if not gws.lib.os2.is_file(path):
            raise gws.base.web.error.NotFound()

        meta = self._read_metadata(path) or self._metadata_from_path(path)
        # @TODO check permissions

        return ReadResponse(data=gws.read_file_b(path))

    def api_delete(self, req: gws.IWebRequest, p: DeleteParams) -> DeleteResponse:
        """Move a file to trash."""

        dp, fname = self._check_file_path(p.path)
        path = dp + '/' + fname

        if not gws.lib.os2.is_file(path):
            raise gws.base.web.error.NotFound()

        meta = self._read_metadata(path) or self._metadata_from_path(path)
        # @TODO check permissions

        os.rename(path, self._trash_path(path))
        meta['deleted'] = True
        self._write_metadata(path, meta)

        return DeleteResponse()

    def api_list(self, req: gws.IWebRequest, p: ListParams) -> ListResponse:
        """Return a list of all server files."""

        entries = []

        for p in gws.lib.os2.find_files(self.root_dir):
            p = gws.lib.os2.rel_path(p, self.root_dir)
            if p.startswith('__'):
                continue
            entries.append(ListEntry(path=p))

        return ListResponse(entries=entries)

    def api_undelete(self, req: gws.IWebRequest, p: UndeleteParams) -> UndeleteResponse:
        """Restore a file from the trash."""

        dp, fname = self._check_file_path(p.path)
        path = dp + '/' + fname
        meta = self._read_metadata(path)

        if not meta or not meta['deleted']:
            raise gws.base.web.error.NotFound()

        os.rename(self._trash_path(path), meta['path'])
        meta['deleted'] = False
        self._write_metadata(path, meta)

        return UndeleteResponse()

    def api_list_trash(self, req: gws.IWebRequest, p: ListParams) -> ListResponse:
        """list paths currently in the trash."""

        entries = []

        with self._connect() as conn:
            rs = conn.execute('SELECT * FROM meta WHERE deleted=1')
            for r in rs:
                entries.append(ListEntry(path=r['path']))

        return ListResponse(entries=entries)

    def api_empty_trash(self, req: gws.IWebRequest, p: EmptyTrashParams) -> EmptyTrashResponse:
        """Empty the trash."""

        with self._connect() as conn:
            conn.execute('DELETE FROM meta WHERE deleted=1')
        for p in gws.lib.os2.find_files(self.trash_dir):
            gws.lib.os2.unlink(p)

        return EmptyTrashResponse()

    ##

    def _check_file_path(self, path):
        pp = gws.lib.os2.parse_path(path)
        dp = gws.lib.os2.abs_path(pp['dirname'], self.root_dir)

        if not dp:
            raise gws.base.web.error.BadRequest('invalid path')

        if not re.match(r'^\w{1,60}', pp['name']):
            raise gws.base.web.error.BadRequest('invalid filename')

        if not re.match(r'^\w{1,60}', pp['extension']):
            raise gws.base.web.error.BadRequest('invalid filename')

        return dp, f"{pp['name']}.{pp['extension']}"

    def _read_metadata(self, path):
        with self._connect() as conn:
            rs = conn.execute('SELECT * FROM meta WHERE path=? LIMIT 1', [path])
            for r in rs:
                return dict(r)

    def _metadata_from_path(self, path):
        if not gws.lib.os2.is_file(path):
            return None
        return {
            'path': path,
            'created_by': 'sys::root',
            'created_time': gws.lib.date.now(),
            'updated_by': 'sys::root',
            'updated_time': gws.lib.date.now(),
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
        gws.lib.os2.unlink(p)
        with self._connect() as conn:
            conn.execute('DELETE FROM meta WHERE path=?', [path])

    def _trash_path(self, path):
        return self.trash_dir + '/' + re.sub(r'\W', '__', path)

    def _connect(self):
        return gws.lib.sqlite.connect(self.db_path)
