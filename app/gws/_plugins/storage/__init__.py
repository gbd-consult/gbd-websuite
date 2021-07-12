"""Storage API."""

import gws.ext.helper.storage

import gws
import gws.types as t
import gws.base.api
import gws.base.web.error


class WriteParams(gws.Params):
    entry: gws.StorageEntry
    data: dict


class WriteResponse(gws.Response):
    entry: gws.StorageEntry


class ReadParams(gws.Params):
    entry: gws.StorageEntry


class ReadResponse(gws.Response):
    entry: gws.StorageEntry
    data: dict


class DirParams(gws.Params):
    category: str


class DirResponse(gws.Response):
    entries: t.List[gws.StorageEntry]
    readable: bool
    writable: bool


class DeleteParams(gws.Params):
    entry: gws.StorageEntry

class DeleteResponse(gws.Params):
    pass


class Config(gws.WithAccess):
    """Storage action"""
    pass


class Object(gws.base.api.Action):

    @gws.cached_property
    def storage(self) -> gws.ext.helper.storage.Object:
        obj: gws.ext.helper.storage.Object = self.root.find_first('gws.ext.helper.storage')
        return obj

    def api_write(self, req: gws.IWebRequest, p: WriteParams) -> WriteResponse:
        try:
            entry = self.storage.write(p.entry, req.user, p.data)
        except gws.ext.helper.storage.AccessDenied:
            raise gws.base.web.error.Forbidden()
        return WriteResponse(entry=entry)

    def api_delete(self, req: gws.IWebRequest, p: DeleteParams) -> DeleteResponse:
        try:
            self.storage.delete(p.entry, req.user)
        except gws.ext.helper.storage.AccessDenied:
            raise gws.base.web.error.Forbidden()
        return DeleteResponse()

    def api_read(self, req: gws.IWebRequest, p: ReadParams) -> ReadResponse:
        try:
            element = self.storage.read(p.entry, req.user)
        except gws.ext.helper.storage.NotFound:
            raise gws.base.web.error.NotFound()
        except gws.ext.helper.storage.AccessDenied:
            raise gws.base.web.error.Forbidden()
        return ReadResponse(element)

    def api_dir(self, req: gws.IWebRequest, p: DirParams) -> DirResponse:
        if not self.storage.can_read_category(p.category, req.user):
            return DirResponse(entries=[], readable=False, writable=False)
        return DirResponse(
            entries=self.storage.dir(p.category, req.user),
            readable=True,
            writable=self.storage.can_write_category(p.category, req.user),
        )
