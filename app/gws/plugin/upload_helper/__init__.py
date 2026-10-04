"""Helper for chunked file uploads.

Large files are uploaded by the client in chunks, one request per chunk. The
helper stores the chunks in the ephemeral directory and, when asked for the
upload, joins them into one file. The helper does not provide endpoints of its
own: an action declares an endpoint that receives the chunks and passes them to
the helper, and another endpoint that processes the finished upload.

Each chunk request contains the file name, the total size, the number of chunks
and the chunk number, starting from 0. The first chunk has an empty
``uploadUid``, which starts a new upload. The handler responds with the
``uploadUid``, which subsequent chunks must provide. Chunks can come in any
order. The total size and the number of chunks are limited by ``maxSize``: the
maximum number of chunks is derived from it, so that a chunk is at least 500 KB
on average.

Once the client has sent all chunks, it calls another endpoint of the action
with the ``uploadUid``. That endpoint calls ``get_upload`` to get the final
file. The file is stored in a temporary location and should be moved to a
permanent location if necessary.

Example::

    helpers+ { type "upload" maxSize 2000 }

Example::

    import gws.plugin.upload_helper as uh


    @gws.ext.command.api('myUpload')
    def do_upload(self, req, p: uh.ChunkRequest) -> uh.ChunkResponse:
        # check permissions, etc...
        helper = self.root.app.helper('upload')
        return helper.handle_chunk_request(req, p)

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


@gws.ext.config.helper('upload')
class Config(gws.Config):
    """Helper that receives chunked file uploads."""

    maxSize: int = 1000
    """Maximum upload size in megabytes."""


class ChunkRequest(gws.Request):
    """Request that carries one chunk of an upload."""

    uploadUid: str = ''
    """Upload uid returned for the first chunk, empty for the first chunk."""
    fileName: str
    """Name of the uploaded file."""
    totalSize: int
    """Total size of the file in bytes."""
    chunkNumber: int
    """Number of this chunk, starting from 0."""
    chunkCount: int
    """Total number of chunks."""
    content: bytes
    """Chunk content."""


class ChunkResponse(gws.Response):
    """Response to a chunk request."""

    uploadUid: str
    """Upload uid, to be passed with subsequent chunks."""


class Upload(gws.Data):
    """State of an upload."""

    uid: str
    """Upload uid."""
    fileName: str
    """Name of the uploaded file, as sent by the client."""
    totalSize: int
    """Total size of the file in bytes."""
    chunkCount: int
    """Total number of chunks."""
    path: str
    """Path of the assembled file, empty until the upload is finalized."""


class Error(gws.Error):
    """Upload error, raised for invalid chunks and incomplete or missing uploads."""

    pass


@gws.ext.object.helper('upload')
class Object(gws.Node):
    """Upload helper, which receives file uploads in chunks and assembles them."""

    maxSize: int
    """Maximum upload size in bytes."""
    maxChunkCount: int
    """Maximum number of chunks per upload."""

    def configure(self):
        self.maxSize = self.cfg('maxSize', default=1000) * 1024 * 1024
        self.maxChunkCount = max(1, self.maxSize // (500 * 1024))  # min. 500K chunks

    def handle_chunk_request(self, req: gws.WebRequester, p: ChunkRequest) -> ChunkResponse:
        """Store a chunk of an upload, starting a new upload if ``uploadUid`` is empty.

        Args:
            req: Web requester.
            p: Chunk request.

        Returns:
            Response with the upload uid.

        Raises:
            ``gws.BadRequestError``: If the chunk or the upload is invalid.
        """
        try:
            up = self._save_chunk(p)
            return ChunkResponse(uploadUid=up.uid)
        except Error as exc:
            gws.log.exception()
            raise gws.BadRequestError('upload_error') from exc

    def get_upload(self, uid: str) -> Upload:
        """Get a finished upload, joining its chunks into one file on the first call.

        Args:
            uid: Upload uid.

        Returns:
            The upload, with ``path`` pointing to the assembled file.

        Raises:
            ``Error``: If the upload is not found, not complete or the file size does not match.
        """
        up = self._load_upload(uid)
        out_path = _base_path(up.uid, 'out')

        if not gws.u.is_file(out_path):
            with gws.u.server_lock(f'upload_{up.uid}'):
                self._finalize(up, out_path)

        up.path = out_path
        return up

    ##

    def _save_chunk(self, p: ChunkRequest) -> Upload:
        """Validate a chunk and write it to the upload directory."""
        up = self._load_upload(p.uploadUid) if p.uploadUid else self._create_upload(p)

        if p.chunkNumber < 0 or p.chunkNumber >= up.chunkCount:
            raise Error(f'upload: {up.uid!r} invalid chunk number')

        if len(p.content) > up.totalSize:
            raise Error(f'upload: {up.uid!r} invalid chunk size')

        with gws.u.server_lock(f'upload_{up.uid}'):
            gws.u.write_file_b(_base_path(up.uid, p.chunkNumber), p.content)

        return up

    def _finalize(self, up: Upload, out_path):
        """Join all chunks into the output file and delete the chunks."""
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
        """Validate the size and chunk count and create a new upload state."""
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
        """Load the state of an existing upload."""
        if not uid.isalnum():
            raise Error(f'upload: invalid uid {uid!r}')
        try:
            return Upload(gws.lib.jsonx.from_path(_base_path(uid, 'state')))
        except gws.lib.jsonx.Error as exc:
            raise Error(f'upload: not found {uid!r}') from exc


def _base_path(uid, p):
    """Return the path of a file in the upload directory."""
    return gws.u.ephemeral_dir(f'upload_{uid}') + f'/{p}'
