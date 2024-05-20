"""Storage object."""

from typing import Optional

import gws
import gws.lib.jsonx


class Verb(gws.Enum):
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
    entryName: Optional[str]
    entryData: Optional[dict]


class Response(gws.Response):
    data: Optional[dict]
    state: State


class Config(gws.ConfigWithAccess):
    """Storage configuration"""

    providerUid: Optional[str]
    """storage provider uid"""

    categoryName: Optional[str]
    """category name"""


class Props(gws.Props):
    state: State


class Object(gws.Node):
    storageProvider: gws.StorageProvider
    categoryName: str

    def configure(self):
        self.configure_provider()
        self.categoryName = self.cfg('categoryName')

    def configure_provider(self):
        self.storageProvider = self.root.app.storageMgr.find_provider(self.cfg('providerUid'))
        if not self.storageProvider:
            raise gws.Error(f'storage provider not found')
        return True

    def props(self, user):
        return gws.Props(
            state=self.get_state_for(user),
        )

    def get_state_for(self, user):
        return State(
            names=self.storageProvider.list_names(self.categoryName) if user.can_read(self) else [],
            canRead=user.can_read(self),
            canWrite=user.can_write(self),
            canDelete=user.can_delete(self),
            canCreate=user.can_create(self),
        )

    def handle_request(self, req: gws.WebRequester, p: Request) -> Response:
        state = self.get_state_for(req.user)
        data = None

        if p.verb == Verb.list:
            if not state.canRead:
                raise gws.ForbiddenError()
            pass

        if p.verb == Verb.read:
            if not state.canRead or not p.entryName:
                raise gws.ForbiddenError()
            rec = self.storageProvider.read(self.categoryName, p.entryName)
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
            self.storageProvider.write(self.categoryName, p.entryName, d, req.user.uid)

        if p.verb == Verb.delete:
            if not state.canDelete or not p.entryName:
                raise gws.ForbiddenError()
            self.storageProvider.delete(self.categoryName, p.entryName)

        return Response(data=data, state=self.get_state_for(req.user))
