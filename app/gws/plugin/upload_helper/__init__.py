"""Manage chunked uploads.

In your action, declare an endpoint with ``p: ChunkRequest` as a parameter. This endpoint should invoke ``handle_chunk_request``::

    import gws.plugin.upload_helper as uh

    @gws.ext.command.api('myUpload')
    def do_upload(self, req, p: uh.ChunkRequest) -> uh.ChunkResponse:
        # check permissions, etc...
        helper = self.root.app.helper('upload')
        return helper.handle_chunk_request(req, p)
        ...

The client sends chunks to this endpoint, one by one. Each chunk contains the file name and total size. The first chunk has an empty ``uploadUid``, indicating a new upload. Subsequent chunks must provide a valid ``uploadUid``. The handler responds with an ``uploadUid``. Each chunk must have a serial number, starting from 0. Chunks can come in any order.

Once the client decides that the upload is complete, it proceeds with invoking some other endpoint of your action, mentioning the ``uploadUid`` returned by the first chunk. The endpoint should invoke ``get_upload`` to retrieve the final file. The file is stored in a temporary location and should be moved to a permanent location if necessary.


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
    uploadUid: str
    fileName: str
    totalSize: int
    path: str


class _Params(gws.Data):
    uid: str
    fileName: str
    totalSize: int
    chunkCount: int


class Error(gws.Error):
    pass


class Object(gws.Node):
    def handle_chunk_request(self, req: gws.WebRequester, p: ChunkRequest) -> ChunkResponse:
        try:
            ps = self._save_chunk(p)
            return ChunkResponse(uploadUid=ps.uid)
        except Error as exc:
            gws.log.exception()
            raise gws.BadRequestError('upload_error') from exc

    def get_upload(self, uid: str) -> Upload:
        ps = self._get_params(uid)
        dd = self._base_dir(ps.uid)
        out_path = f'{dd}/out'
        if not gws.u.is_file(out_path):
            with gws.u.server_lock(f'upload_{ps.uid}'):
                self._finalize(ps, out_path)
        return Upload(uploadUid=ps.uid, path=out_path, fileName=ps.fileName)

    ##

    def _save_chunk(self, p: ChunkRequest) -> _Params:
        ps = self._get_params(p.uploadUid) if p.uploadUid else self._create_upload(p)

        if p.chunkNumber < 0 or p.chunkNumber >= ps.chunkCount:
            raise Error(f'upload: {ps.uid!r} invalid chunk number')

        dd = self._base_dir(ps.uid)

        with gws.u.server_lock(f'upload_{ps.uid}'):
            gws.u.write_file_b(f'{dd}/{p.chunkNumber}', p.content)

        return ps

    def _get_chunks(self, ps: _Params):
        dd = self._base_dir(ps.uid)
        chunks = [f'{dd}/{n}' for n in range(0, ps.chunkCount)]
        complete = all(gws.u.is_file(c) for c in chunks)
        return chunks, complete

    def _finalize(self, ps: _Params, out_path):
        chunks, complete = self._get_chunks(ps)
        if not complete:
            raise Error(f'upload: {ps.uid!r} incomplete')

        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'wb') as fp_all:
            for c in chunks:
                try:
                    with open(c, 'rb') as fp:
                        shutil.copyfileobj(fp, fp_all)
                except (OSError, IOError) as exc:
                    raise Error(f'upload: {ps.uid!r}: IO error') from exc

        if gws.lib.osx.file_size(tmp_path) != ps.totalSize:
            raise Error(f'upload: {ps.uid!r}: invalid file size')

        # @TODO check checksums as well?

        try:
            gws.lib.osx.rename(tmp_path, out_path)
        except OSError:
            raise Error(f'upload: {ps.uid!r}: move error')

        for c in chunks:
            gws.lib.osx.unlink(c)

    def _create_upload(self, p: ChunkRequest) -> _Params:
        uid = gws.u.random_string(64)
        dd = self._base_dir(uid)

        gws.lib.jsonx.to_path(f'{dd}/s.json', _Params(
            uid=uid,
            fileName=p.fileName,
            totalSize=p.totalSize,
            chunkCount=p.chunkCount,
        ))

        return self._get_params(uid)

    def _get_params(self, uid):
        if not uid.isalnum():
            raise Error(f'upload: {uid!r} invalid')

        dd = self._base_dir(uid)

        try:
            return _Params(gws.lib.jsonx.from_path(f'{dd}/s.json'))
        except gws.lib.jsonx.Error as exc:
            raise Error(f'upload: {uid!r} not found') from exc

    def _base_dir(self, uid):
        return gws.u.ephemeral_dir(f'upload_{uid}')
