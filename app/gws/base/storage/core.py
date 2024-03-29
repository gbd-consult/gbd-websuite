"""Storage object."""

import gws
import gws.lib.jsonx
import gws.types as t


class Verb(t.Enum):
    read = 'read'
    write = 'write'
    list = 'list'
    delete = 'delete'


class State(gws.Data):
    names: list[str]
    canRead: bool
    canWrite: bool
    canCreate: bool
    canDelete: bool


class Request(gws.Request):
    verb: Verb
    entryName: t.Optional[str]
    entryData: t.Optional[dict]


class Response(gws.Response):
    data: t.Optional[dict]
    state: State


class Config(gws.ConfigWithAccess):
    """Storage configuation"""

    providerUid: t.Optional[str]
    """storage provider uid"""

    categoryName: t.Optional[str]
    """category name"""


class Props(gws.Props):
    state: State


class Object(gws.Node):
    provider: gws.IStorageProvider
    categoryName: str

    def configure(self):
        self.configure_provider()
        self.categoryName = self.cfg('categoryName')

    def configure_provider(self):
        mgr = self.root.app.storageMgr

        uid = self.cfg('providerUid')
        if uid:
            self.provider = mgr.provider(uid)
            if not self.provider:
                raise gws.Error(f'storage provider {uid!r} not found')
            return True

        self.provider = mgr.first_provider()
        if not self.provider:
            raise gws.Error(f'no storage providers configured')
        return True

    def props(self, user):
        return gws.Props(
            state=self.get_state_for(user),
        )

    def get_state_for(self, user):
        return State(
            names=self.provider.list_names(self.categoryName) if user.can_read(self) else [],
            canRead=user.can_read(self),
            canWrite=user.can_write(self),
            canDelete=user.can_delete(self),
            canCreate=user.can_create(self),
        )

    def handle_request(self, req: gws.IWebRequester, p: Request) -> Response:
        state = self.get_state_for(req.user)
        data = None

        if p.verb == Verb.list:
            if not state.canRead:
                raise gws.ForbiddenError()
            pass

        if p.verb == Verb.read:
            if not state.canRead or not p.entryName:
                raise gws.ForbiddenError()
            rec = self.provider.read(self.categoryName, p.entryName)
            if rec:
                data = gws.lib.jsonx.from_string(rec.data)

        if p.verb == Verb.write:
            if not p.entryName or not p.entryData:
                raise gws.ForbiddenError()
            if p.entryName in state.names and not state.canWrite:
                raise gws.ForbiddenError()
            if p.entryName not in state.names and not state.canCreate:
                raise gws.ForbiddenError()

            d = gws.lib.jsonx.to_string(p.entryData)
            self.provider.write(self.categoryName, p.entryName, d, req.user.uid)

        if p.verb == Verb.delete:
            if not state.canDelete or not p.entryName:
                raise gws.ForbiddenError()
            self.provider.delete(self.categoryName, p.entryName)

        return Response(data=data, state=self.get_state_for(req.user))
