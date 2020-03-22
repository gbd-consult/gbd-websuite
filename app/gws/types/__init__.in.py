from typing import Any, Dict, List, Optional, Tuple, Union, cast


# NB: we cannot use the standard Enum, because after "class Color(Enum): RED = 1"
# the value of Color.RED is like {'_value_': 1, '_name_': 'RED', '__objclass__': etc}
# and we need it to be 1, literally (that's what we'll get from the client)

class Enum:
    pass


def none() -> Any:
    return None


#:alias An array of 4 elements representing extent coordinates [minx, miny, maxx, maxy]
Extent = Tuple[float, float, float, float]

#:alias Point coordinates [x, y]
Point = Tuple[float, float]

#:alias Size [width, height]
Size = Tuple[float, float]

#:alias A value with a unit
Measurement = Tuple[float, str]


class Axis(Enum):
    """Axis orientation."""
    xy = 'xy'
    yx = 'yx'


#:alias Verbatim literal type
Literal = str

#:alias Valid readable file path on the server
FilePath = str

#:alias Valid readable directory path on the server
DirPath = str

#:alias String like "1w 2d 3h 4m 5s" or a number of seconds
Duration = str

#:alias CSS color name
Color = str

#:alias Regular expression, as used in Python
Regex = str

#:alias String with {attribute} placeholders
FormatStr = str

#:alias CRS code like "EPSG:3857
Crs = str

#:alias ISO date like "2019-01-30"
Date = str

#:alias Http or https URL
Url = str


# dummy classes to support extension typing

class ext:
    class action:
        class Config:
            pass

        class Props:
            pass

    class auth:
        class method:
            class Config:
                pass

        class provider:
            class Config:
                pass

    class template:
        class Config:
            pass

        class Props:
            pass

    class db:
        class provider:
            class Config:
                pass

    class layer:
        class Config:
            pass

        class Props:
            pass

    class search:
        class provider:
            class Config:
                pass

    class storage:
        class Config:
            pass

    class helper:
        class Config:
            pass

    class ows:
        class provider:
            class Config:
                pass

        class service:
            class Config:
                pass


# basic data type

class Data:
    """Basic data object."""

    def __init__(self, *args, **kwargs):
        self._extend(args, kwargs)

    def __repr__(self):
        return repr(vars(self))

    def __getattr__(self, item):
        if item.startswith('_'):
            # do not use None fallback for special props
            raise AttributeError()
        return None

    def get(self, k, default=None):
        return vars(self).get(k, default)

    def _extend(self, args, kwargs):
        d = {}
        for a in args:
            if isinstance(a, dict):
                d.update(a)
            elif isinstance(a, Data):
                d.update(vars(a))
        d.update(kwargs)
        vars(self).update(d)


# configuration primitives

class Config(Data):
    """Configuration base type"""

    uid: str = ''  #: unique ID


class WithType(Config):
    type: str  #: object type


class AccessType(Enum):
    allow = 'allow'
    deny = 'deny'


class Access(Config):
    """Access rights definition for authorization roles"""

    type: AccessType  #: access type (deny or allow)
    role: str  #: a role to which this rule applies


class WithAccess(Config):
    access: Optional[List[Access]]  #: access rights


class WithTypeAndAccess(Config):
    type: str  #: object type
    access: Optional[List[Access]]  #: access rights


# attributes

class AttributeType(Enum):
    bool = 'bool'
    bytes = 'bytes'
    date = 'date'
    datetime = 'datetime'
    float = 'float'
    geometry = 'geometry'
    int = 'int'
    list = 'list'
    str = 'str'
    text = 'text'
    time = 'time'


class GeometryType(Enum):
    curve = 'CURVE'
    geomcollection = 'GEOMCOLLECTION'
    geometry = 'GEOMETRY'
    linestring = 'LINESTRING'
    multicurve = 'MULTICURVE'
    multilinestring = 'MULTILINESTRING'
    multipoint = 'MULTIPOINT'
    multipolygon = 'MULTIPOLYGON'
    multisurface = 'MULTISURFACE'
    point = 'POINT'
    polygon = 'POLYGON'
    polyhedralsurface = 'POLYHEDRALSURFACE'
    surface = 'SURFACE'


class Attribute(Data):
    name: str
    title: str = ''
    type: AttributeType = 'str'
    value: Any = None
    editable: bool = True


# request params and responses

class Params(Data):
    projectUid: Optional[str]  #: project uid
    locale: Optional[str]  #: locale for this request


class NoParams(Data):
    pass


class ResponseError(Data):
    status: int
    info: str


class Response(Data):
    error: Optional[ResponseError]


class HttpResponse(Response):
    mime: str
    content: str
    status: int


class FileResponse(Response):
    mime: str
    path: str
    status: int
    attachment_name: str


# props baseclass


class Props(Data):
    """Properties base type"""
    pass
