import gws.tools.xml2
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


def read(text, crs=None, invert_axis=None, **kwargs) -> t.List[t.IFeature]:
    first_el = gws.tools.xml2.peek(text)

    if first_el:
        # remove the xml declaration, in order not to confuse gdal with non-utf8 encodings
        text = gws.tools.xml2.strip_before(text, first_el)
        for p in xml_formats:
            res = p.parse(text, first_el, crs, invert_axis, **kwargs)
            if res is not None:
                gws.log.debug(f'parsed with {p!r}')
                return res

    else:

        for p in text_formats:
            res = p.parse(text, **kwargs)
            if res is not None:
                gws.log.debug(f'parsed with {p!r}')
                return res
