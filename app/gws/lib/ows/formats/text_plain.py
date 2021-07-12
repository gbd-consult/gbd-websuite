import gws.lib.feature
import gws.lib.shape
import gws.lib.xml2


# plain text format

def parse(s, **kwargs):
    return [gws.lib.feature.Feature(attributes={'text': s})]
