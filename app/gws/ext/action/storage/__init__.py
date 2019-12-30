import gws
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
    pass


class DirResponse(t.Response):
    entries: t.List[t.StorageEntry]


class Config(t.WithTypeAndAccess):
    """Storage action"""
    pass


class Object(gws.ActionObject):

    @property
    def storage(self) -> t.IStorage:
        return self.root.application.storage

    def api_write(self, req: t.IRequest, p: WriteParams) -> WriteResponse:
        ok = self.storage.write(p.entry, req.user, p.data)
        if not ok:
            raise gws.web.error.Forbidden()
        return WriteResponse({
            'entry': p.entry
        })

    def api_read(self, req: t.IRequest, p: ReadParams) -> ReadResponse:
        return ReadResponse({
            'entry': p.entry,
            'data': self.storage.read(p.entry, req.user)
        })

    def api_dir(self, req: t.IRequest, p: DirParams) -> DirResponse:
        return DirResponse({
            'entries': self.storage.dir(req.user)
        })
