import gws
import gws.types as t


class PermissionMode(t.Enum):
    read = 'read'  #: an object can be read
    write = 'write'  #: an object can be written and deleted
    all = 'all'  #: an object can be read and written


class PermissionRule(gws.WithAccess):
    """Permission rule for a storage category"""

    category: str  #: storage category name
    mode: PermissionMode  #: allowed mode (read/write)


# @TODO more props, like author, time etc

class Entry(gws.Data):
    name: str


class Directory(gws.Data):
    category: str
    writable: bool
    readable: bool
    entries: t.List[Entry]


class Record(gws.Data):
    category: str
    name: str
    user_fid: str
    data: str
    created: int
    updated: int


class Verb(t.Enum):
    read = 'read'
    write = 'write'
    list = 'list'
    delete = 'delete'


class Params(gws.Params):
    verb: Verb
    entryName: t.Optional[str]
    entryData: t.Optional[dict]


class Response(gws.Response):
    directory: Directory
    data: dict
