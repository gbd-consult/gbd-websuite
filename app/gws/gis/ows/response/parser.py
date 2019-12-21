import gws.tools.xml3
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


def parse(s, **kwargs) -> t.List[t.Feature]:
    first_el = gws.tools.xml3.peek(s)

    if first_el:
        # remove xml declaration, in order not to confuse gdal with non-utf8 encodings
        s = gws.tools.xml3.strip_before(s, first_el)
        for p in xml_formats:
            res = p.parse(s, first_el, **kwargs)
            if res is not None:
                gws.log.debug(f'parsed with {p!r}')
                return res

    else:

        for p in text_formats:
            res = p.parse(s, **kwargs)
            if res is not None:
                gws.log.debug(f'parsed with {p!r}')
                return res
