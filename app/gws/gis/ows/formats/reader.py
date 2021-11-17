import gws
import gws.lib.xml2
import gws.types as t

from . import (

    feature_collection,
    feature_info_response,
    geobak,
    get_feature_info_response,
    text_plain,
)

xml_formats = [
    feature_collection,
    feature_info_response,
    geobak,
    get_feature_info_response,
]

text_formats = [
    text_plain,
]


def read(text, crs: t.Optional[gws.ICrs] = None, axis: t.Optional[gws.Axis] = gws.AXIS_XY, **kwargs) -> t.List[gws.IFeature]:
    first_el = gws.lib.xml2.peek(text)

    invert_axis = axis == gws.AXIS_YX

    if first_el:
        # remove the xml declaration, in order not to confuse gdal with non-utf8 encodings
        text = gws.lib.xml2.strip_before(text, first_el)
        for mod in xml_formats:
            fn = getattr(mod, 'parse')
            res = fn(text, first_el, crs, invert_axis, **kwargs)
            if res is not None:
                gws.log.debug(f'parsed with {mod!r}')
                return res

    else:
        for mod in text_formats:
            fn = getattr(mod, 'parse')
            res = fn(text, crs, invert_axis, **kwargs)
            if res is not None:
                gws.log.debug(f'parsed with {mod!r}')
                return res
