### Basic types

# noinspection PyUnresolvedReferences
from typing import Optional, List, Dict, Tuple, Union, Any, cast


# NB: we cannot use the standard Enum, because after "class Color(Enum): RED = 1"
# the value of Color.RED is like {'_value_': 1, '_name_': 'RED', '__objclass__': etc}
# and we need it to be 1, literally (that's what we'll get from the client)

class Enum:
    pass


#:alias An array of 4 elements representing extent coordinates [minx, miny, maxx, maxy]
Extent = Tuple[float, float, float, float]

#:alias Point coordinates [x, y]
Point = Tuple[float, float]

#:alias Size [width, height]
Size = Tuple[float, float]


class Axis(Enum):
    xy = 'xy'
    yx = 'yx'


### semantic primitive types

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
