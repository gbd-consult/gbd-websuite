import gws
import gws.base.web.error
import gws.lib.jsonx
import gws.types as t


class CategoryConfig(gws.ConfigWithAccess):
    """Storage category"""
    name: str
    """category name"""


class Verb(t.Enum):
    read = 'read'
    write = 'write'
    list = 'list'
    delete = 'delete'


class Request(gws.Request):
    verb: Verb
    entryName: t.Optional[str]
    entryData: t.Optional[dict]


class State(gws.Data):
    names: list[str]
    canRead: bool
    canWrite: bool
    canCreate: bool
    canDelete: bool


class Response(gws.Response):
    data: t.Optional[dict]
    state: State
