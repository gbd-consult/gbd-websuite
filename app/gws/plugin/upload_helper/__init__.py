"""Manage chunked uploads.

In your action, declare an endpoint with ``p: ChunkRequest`` as a parameter. This endpoint should invoke ``handle_chunk_request``::

    import gws.plugin.upload_helper as uh


    @gws.ext.command.api('myUpload')
    def do_upload(self, req, p: uh.ChunkRequest) -> uh.ChunkResponse:
        # check permissions, etc...
        helper = self.root.app.helper('upload')
        return helper.handle_chunk_request(req, p)
        ...

The client sends chunks to this endpoint, one by one. Each chunk contains the file name and total size. The first chunk has an empty ``uploadUid``, indicating a new upload. Subsequent chunks must provide a valid ``uploadUid``. The handler responds with an ``uploadUid``. Each chunk must have a serial number, starting from 0. Chunks can come in any order.

Once the client decides that the upload is complete, it proceeds with invoking some other endpoint of your action, mentioning the ``uploadUid`` returned by the first chunk. The endpoint should invoke ``get_upload`` to retrieve the final file. The file is stored in a temporary location and should be moved to a permanent location if necessary::

    @gws.ext.command.api('myProcessUploadedFile')
    def do_process(self, req, p: MyProcessRequest):
        helper = self.root.app.helper('upload')
        try:
            upload = helper.get_upload(p.uploadUid)
        except uh.Error:
            ...upload not ready yet...
        ...process(upload.path)



"""

import shutil

import gws
import gws.lib.jsonx
import gws.lib.osx

gws.ext.new.helper('upload')


class Config(gws.Config):
    """Upload helper."""

    maxSize: int = 1000
    """Maximum upload size in megabytes."""


class ChunkRequest(gws.Request):
    uploadUid: str = ''
    fileName: str
    totalSize: int
    chunkNumber: int
    chunkCount: int
    content: bytes


class ChunkResponse(gws.Response):
    uploadUid: str


class Upload(gws.Data):
    uid: str
    fileName: str
    totalSize: int
    chunkCount: int
    path: str


class Error(gws.Error):
    pass


class Object(gws.Node):
    maxSize: int
    maxChunkCount: int

    def configure(self):
        self.maxSize = self.cfg('maxSize', default=1000) * 1024 * 1024
        self.maxChunkCount = max(1, self.maxSize // (500 * 1024))  # min. 500K chunks

    def handle_chunk_request(self, req: gws.WebRequester, p: ChunkRequest) -> ChunkResponse:
        try:
            up = self._save_chunk(p)
            return ChunkResponse(uploadUid=up.uid)
        except Error as exc:
            gws.log.exception()
            raise gws.BadRequestError('upload_error') from exc

    def get_upload(self, uid: str) -> Upload:
        up = self._load_upload(uid)
        out_path = _base_path(up.uid, 'out')

        if not gws.u.is_file(out_path):
            with gws.u.server_lock(f'upload_{up.uid}'):
                self._finalize(up, out_path)

        up.path = out_path
        return up

    ##

    def _save_chunk(self, p: ChunkRequest) -> Upload:
        up = self._load_upload(p.uploadUid) if p.uploadUid else self._create_upload(p)

        if p.chunkNumber < 0 or p.chunkNumber >= up.chunkCount:
            raise Error(f'upload: {up.uid!r} invalid chunk number')

        if len(p.content) > up.totalSize:
            raise Error(f'upload: {up.uid!r} invalid chunk size')

        with gws.u.server_lock(f'upload_{up.uid}'):
            gws.u.write_file_b(_base_path(up.uid, p.chunkNumber), p.content)

        return up

    def _finalize(self, up: Upload, out_path):
        chunks = [_base_path(up.uid, n) for n in range(0, up.chunkCount)]
        complete = all(gws.u.is_file(c) for c in chunks)
        if not complete:
            raise Error(f'upload: {up.uid!r}: incomplete')

        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'wb') as fp_all:
            for c in chunks:
                try:
                    with open(c, 'rb') as fp:
                        shutil.copyfileobj(fp, fp_all)
                except (OSError, IOError) as exc:
                    raise Error(f'upload: {up.uid!r}: IO error') from exc

        if gws.lib.osx.file_size(tmp_path) != up.totalSize:
            raise Error(f'upload: {up.uid!r}: invalid file size')

        # @TODO check checksums as well?

        try:
            gws.lib.osx.rename(tmp_path, out_path)
        except OSError:
            raise Error(f'upload: {up.uid!r}: move error')

        for c in chunks:
            gws.lib.osx.unlink(c)

    def _create_upload(self, p: ChunkRequest) -> Upload:
        if p.totalSize <= 0 or p.totalSize > self.maxSize:
            raise Error(f'upload: invalid total size {p.totalSize!r}')
        if p.chunkCount <= 0 or p.chunkCount > self.maxChunkCount:
            raise Error(f'upload: invalid chunk count {p.chunkCount!r}')

        uid = gws.u.random_string(64)
        up = Upload(
            uid=uid,
            fileName=p.fileName,
            totalSize=p.totalSize,
            chunkCount=p.chunkCount,
            path='',
        )
        gws.lib.jsonx.to_path(_base_path(uid, 'state'), up)
        return up

    def _load_upload(self, uid) -> Upload:
        if not uid.isalnum():
            raise Error(f'upload: invalid uid {uid!r}')
        try:
            return Upload(gws.lib.jsonx.from_path(_base_path(uid, 'state')))
        except gws.lib.jsonx.Error as exc:
            raise Error(f'upload: not found {uid!r}') from exc


def _base_path(uid, p):
    return gws.u.ephemeral_dir(f'upload_{uid}') + f'/{p}'
