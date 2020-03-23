"""Handles chunked uploads."""

import gws.common.action
import gws.tools.upload
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Chunked upload action"""
    pass


class ChunkParams(t.Params):
    uid: str
    name: str
    totalSize: int
    content: bytes
    chunkNumber: int
    chunkCount: int


class ChunkResponse(t.Response):
    uid: str


class Object(gws.common.action.Object):
    def api_send_chunk(self, req: t.IRequest, p: ChunkParams) -> ChunkResponse:
        """Receive one chunk"""

        try:
            uid = gws.tools.upload.save_chunk(
                uid=p.uid,
                name=p.name,
                total_size=p.totalSize,
                content=p.content,
                chunk_count=p.chunkCount,
                chunk_number=p.chunkNumber,
            )
            return ChunkResponse(uid=uid)
        except gws.tools.upload.Error as e:
            gws.log.error(e)
            raise gws.web.error.BadRequest()
