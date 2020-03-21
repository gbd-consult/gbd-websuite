"""Storage API."""

import gws
import gws.common.action
import gws.ext.helper.storage
import gws.web.error

import gws.types as t


class WriteParams(t.Params):
    entry: t.StorageEntry
    data: dict


class WriteResponse(t.Response):
    entry: t.StorageEntry


class ReadParams(t.Params):
    entry: t.StorageEntry


class ReadResponse(t.Response):
    entry: t.StorageEntry
    data: dict


class DirParams(t.Params):
    category: str


class DirResponse(t.Response):
    entries: t.List[t.StorageEntry]
    readable: bool
    writable: bool


class DeleteParams(t.Params):
    entry: t.StorageEntry

class DeleteResponse(t.Params):
    pass


class Config(t.WithTypeAndAccess):
    """Storage action"""
    pass


class Object(gws.common.action.Object):

    @gws.cached_property
    def storage(self) -> gws.ext.helper.storage.Object:
        obj: gws.ext.helper.storage.Object = self.root.find_first('gws.ext.helper.storage')
        return obj

    def api_write(self, req: t.IRequest, p: WriteParams) -> WriteResponse:
        try:
            entry = self.storage.write(p.entry, req.user, p.data)
        except gws.ext.helper.storage.AccessDenied:
            raise gws.web.error.Forbidden()
        return WriteResponse(entry=entry)

    def api_delete(self, req: t.IRequest, p: DeleteParams) -> DeleteResponse:
        try:
            self.storage.delete(p.entry, req.user)
        except gws.ext.helper.storage.AccessDenied:
            raise gws.web.error.Forbidden()
        return DeleteResponse()

    def api_read(self, req: t.IRequest, p: ReadParams) -> ReadResponse:
        try:
            element = self.storage.read(p.entry, req.user)
        except gws.ext.helper.storage.NotFound:
            raise gws.web.error.NotFound()
        except gws.ext.helper.storage.AccessDenied:
            raise gws.web.error.Forbidden()
        return ReadResponse(element)

    def api_dir(self, req: t.IRequest, p: DirParams) -> DirResponse:
        if not self.storage.can_read_category(p.category, req.user):
            return DirResponse(entries=[], readable=False, writable=False)
        return DirResponse(
            entries=self.storage.dir(p.category, req.user),
            readable=True,
            writable=self.storage.can_write_category(p.category, req.user),
        )
