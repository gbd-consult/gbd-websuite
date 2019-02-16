import re

import gws
import gws.types as t

"""

NB: layer order
our configuration lists layers top-to-bottom,
this also applies by default to WMS caps (like in qgis)

the order of GetMap is bottomUp:

> A WMS shall render the requested layers by drawing the leftmost in the list bottommost, 
> the next one over that, and so on.

http://portal.opengeospatial.org/files/?artifact_id=14416
section 7.3.3.3 

"""


class LayerOrder(t.Enum):
    topDown = 'topDown'
    bottomUp = 'bottomUp'


class LayerFilterConfig(t.Config):
    """Layer filter"""

    level: int = 0  #: use layers at this level
    names: t.Optional[t.List[str]]  #: use these layer names
    pattern: t.regex = ''  #: match a pattern against the layer full path


class LayerFilter(t.Data):
    level: int
    names: t.List[str]
    pattern: str


def filter_layers(
        layers: t.List[t.SourceLayer],
        slf: LayerFilter,
        image_only=False,
        queryable_only=False) -> t.List[t.SourceLayer]:
    if slf:
        s = gws.get(slf, 'level')
        if s:
            layers = [sl for sl in layers if sl.a_level == s]

        s = gws.get(slf, 'names')
        if s:
            layers = [sl for sl in layers if sl.name in s]

        s = gws.get(slf, 'pattern')
        if s:
            layers = [sl for sl in layers if re.search(s, sl.a_path)]

    if image_only:
        layers = [sl for sl in layers if sl.is_image]
    if queryable_only:
        layers = [sl for sl in layers if sl.is_queryable]

    return layers
