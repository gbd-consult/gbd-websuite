import gws.tools.xml2
import gws.gis.shape
import gws.common.model


# plain text format

def parse(s, **kwargs):
    return [gws.common.model.generic_feature(attributes={'text': s})]
