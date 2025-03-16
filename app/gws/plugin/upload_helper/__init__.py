"""Manage chunked uploads.

In your action, declare an endpoint with ``p: ChunkRequest` as a parameter. This endpoint should invoke ``handle_chunk_request``::

    import gws.plugin.upload_helper as uh

    @gws.ext.command.api('myUpload')
    def do_upload(self, req, p: uh.ChunkRequest) -> uh.ChunkResponse:
        # check permissions, etc...
        helper = self.root.app.helper('upload')
        return helper.handle_chunk_request(req, p)
        ...

The client sends chunks to this endpoint, one by one. Each chunk contains the file name and total size. The first chunk has an empty ``uploadUid``, indicating a new upload. Subsequent chunks must provide a valid ``uploadUid``. The handler responds with an ``uploadUid`` and a flag indicating whether the upload is complete. Each chunk must have a serial number, starting from 0. Chunks can come in any order.

Once the client decides that the upload is complete, it proceeds with invoking some other endpoint of your action, mentioning the ``uploadUid`` returned by the first chunk. The endpoint should invoke ``get_upload`` to retrieve the final file. The file is stored in a temporary location and should be moved to a permanent location if necessary.


    @gws.ext.command.api('myProcessUploadedFile')
    def do_upload(self, req, p: MyProcessRequest):
        helper = self.root.app.helper('upload')
        upload = helper.get_upload(p.uploadUid)
        if not upload.path:
            raise Error('upload not ready')
        ...process `upload.path`



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
    isComplete: bool


class Upload(gws.Data):
    uploadUid: str
    fileName: str
    totalSize: int
    path: str


class Status(gws.Data):
    uploadUid: str
    fileName: str
    totalSize: int
    chunkCount: int


class Error(gws.Error):
    pass


class Object(gws.Node):
    def handle_chunk_request(self, req: gws.WebRequester, p: ChunkRequest) -> ChunkResponse:
        try:
            st = self._save_chunk(p)
            return ChunkResponse(uploadUid=st.uploadUid, isComplete=self._is_complete(st))
        except Error as exc:
            gws.log.exception()
            raise gws.BadRequestError('upload_error') from exc

    def _save_chunk(self, p: ChunkRequest) -> Status:
        dd = self._base_dir()

        if not p.uploadUid:
            p.uploadUid = gws.u.random_string(64)
            gws.lib.jsonx.to_path(f'{dd}/{p.uploadUid}.json', Status(
                uploadUid=p.uploadUid,
                fileName=p.fileName,
                totalSize=p.totalSize,
                chunkCount=p.chunkCount,
            ))

        # @TODO check if fixed params haven't been changed in subsequent chunks

        st = self._get_status(p.uploadUid)
        if p.chunkNumber < 0 or p.chunkNumber >= st.chunkCount:
            raise Error(f'upload: {p.uploadUid!r} invalid chunk number')

        with gws.u.server_lock(f'upload_{st.uploadUid}'):
            gws.u.write_file_b(f'{dd}/{st.uploadUid}.{p.chunkNumber}', p.content)

        return st

    def get_upload(self, uid: str) -> Upload:
        dd = self._base_dir()
        st = self._get_status(uid)
        out_path = f'{dd}/{uid}.out'
        if not gws.u.is_file(out_path):
            self._finalize(st, out_path)
        return Upload(uploadUid=st.uploadUid, path=out_path, fileName=st.fileName)

    ##

    def _is_complete(self, st: Status):
        dd = self._base_dir()
        chunks = [f'{dd}/{st.uploadUid}.{n}' for n in range(0, st.chunkCount)]
        return all(gws.u.is_file(c) for c in chunks)

    def _finalize(self, st: Status, out_path):
        dd = self._base_dir()

        chunks = [f'{dd}/{st.uploadUid}.{n}' for n in range(0, st.chunkCount)]
        if not all(gws.u.is_file(c) for c in chunks):
            raise Error(f'upload: {st.uploadUid!r} incomplete')

        with gws.u.server_lock(f'upload_{st.uploadUid}'):
            chunks = [f'{dd}/{st.uploadUid}.{n}' for n in range(0, st.chunkCount)]
            if not all(gws.u.is_file(c) for c in chunks):
                raise Error(f'upload: {st.uploadUid!r} incomplete')

            tmp_path = out_path + '.tmp'
            with open(tmp_path, 'wb') as fp_all:
                for c in chunks:
                    try:
                        with open(c, 'rb') as fp:
                            shutil.copyfileobj(fp, fp_all)
                    except (OSError, IOError) as exc:
                        raise Error(f'upload: {st.uploadUid!r}: IO error') from exc

            if gws.lib.osx.file_size(tmp_path) != st.totalSize:
                raise Error(f'upload: {st.uploadUid!r}: invalid file size')

            # @TODO check checksums as well?

            try:
                gws.lib.osx.rename(tmp_path, out_path)
            except OSError:
                raise Error(f'upload: {st.uploadUid!r}: move error')

            for c in chunks:
                gws.lib.osx.unlink(c)

    def _get_status(self, uid):
        dd = self._base_dir()

        if not uid.isalnum():
            raise Error(f'upload: {uid!r} invalid')

        try:
            return Status(gws.lib.jsonx.from_path(f'{dd}/{uid}.json'))
        except gws.lib.jsonx.Error as exc:
            raise Error(f'upload: {uid!r} not found') from exc

    def _base_dir(self):
        return gws.u.ephemeral_dir('uploads')
