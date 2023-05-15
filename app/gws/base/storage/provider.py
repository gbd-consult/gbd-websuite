"""Base storage provider."""

import gws
import gws.base.web
import gws.lib.date
import gws.lib.jsonx
import gws.types as t

from . import core


class Category(gws.Node):
    name: str

    def configure(self):
        self.name = self.cfg('name')


class Record(gws.Data):
    name: str
    userUid: str
    data: str
    created: int
    updated: int


class Config(gws.ConfigWithAccess):
    categories: t.Optional[list[core.CategoryConfig]]


class Object(gws.Node):
    categories: dict[str, Category]

    def configure(self):
        self.categories = {}

        for p in self.cfg('categories', default=[]):
            self.add_category(p)

        if '*' not in self.categories:
            self.categories['*'] = self.create_child(Category, name='*')

    def add_category(self, p: core.CategoryConfig):
        cat = self.create_child(Category, p)
        self.categories[cat.name] = cat
        return cat.name

    def get_state(self, category_name: str, user: gws.IUser) -> core.State:
        cat = self.categories.get(category_name) or self.categories.get('*')
        state = core.State(
            canRead=user.can_read(cat),
            canWrite=user.can_write(cat),
            canDelete=user.can_delete(cat),
            canCreate=user.can_create(cat),
        )
        state.names = self.db_list_names(cat) if state.canRead else []
        return state

    def handle_request(self, category_name: str, req: gws.IWebRequester, p: core.Request) -> core.Response:
        cat = self.categories.get(category_name) or self.categories.get('*')

        can_read = req.user.can_read(cat)
        can_write = req.user.can_write(cat)
        can_delete = req.user.can_delete(cat)
        can_create = req.user.can_create(cat)

        data = None
        names = self.db_list_names(cat)

        if p.verb == core.Verb.list:
            if not can_read:
                raise gws.base.web.error.Forbidden()
            pass

        if p.verb == core.Verb.read:
            if not can_read or not p.entryName:
                raise gws.base.web.error.Forbidden()
            rec = self.db_read(cat, p.entryName)
            if rec:
                data = gws.lib.jsonx.from_string(rec.data)

        if p.verb == core.Verb.write:
            if not p.entryName or not p.entryData:
                raise gws.base.web.error.Forbidden()

            if p.entryName in names and not can_write:
                raise gws.base.web.error.Forbidden()
            if p.entryName not in names and not can_create:
                raise gws.base.web.error.Forbidden()

            d = gws.lib.jsonx.to_string(p.entryData)
            self.db_write(cat, p.entryName, d, req.user.uid)

        if p.verb == core.Verb.delete:
            if not can_delete or not p.entryName:
                raise gws.base.web.error.Forbidden()
            self.db_delete(cat, p.entryName)

        state = core.State(
            names=names if can_read else [],
            canRead=can_read,
            canWrite=can_write,
            canDelete=can_delete,
            canCreate=can_create,
        )

        return core.Response(data=data, state=state)

    def db_list_names(self, cat: Category) -> list[str]:
        pass

    def db_read(self, cat: Category, name: str) -> t.Optional[Record]:
        pass

    def db_write(self, cat: Category, name: str, data: str, user_uid: str):
        pass

    def db_delete(self, cat: Category, name: str):
        pass


##

def get_for(obj: gws.INode, uid: str = ''):
    mgr = obj.root.app.storageMgr

    uid = uid or obj.cfg('storageUid')
    if uid:
        p = mgr.provider(uid)
        if not p:
            raise gws.Error(f'storage provider {uid!r} not found')
        return p

    p = mgr.first_provider()
    return p
