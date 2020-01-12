import gws
import gws.web.error
import gws.ext.tool.storage

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


class Config(t.WithTypeAndAccess):
    """Storage action"""
    pass


class Object(gws.ActionObject):

    @gws.cached_property
    def storage(self) -> gws.ext.tool.storage.Object:
        obj: gws.ext.tool.storage.Object = self.find_first('gws.ext.tool.storage')
        return obj

    def api_write(self, req: t.IRequest, p: WriteParams) -> WriteResponse:
        try:
            entry = self.storage.write(p.entry, req.user, p.data)
        except gws.ext.tool.storage.AccessDenied:
            raise gws.web.error.Forbidden()
        return WriteResponse(entry=entry)

    def api_read(self, req: t.IRequest, p: ReadParams) -> ReadResponse:
        try:
            element = self.storage.read(p.entry, req.user)
        except gws.ext.tool.storage.NotFound:
            raise gws.web.error.NotFound()
        except gws.ext.tool.storage.AccessDenied:
            raise gws.web.error.Forbidden()
        return ReadResponse(element)

    def api_dir(self, req: t.IRequest, p: DirParams) -> DirResponse:
        entries = self.storage.dir(p.category, req.user)
        return DirResponse(entries=entries)
