import gws.tools.xml3
import gws.gis.shape
import gws.gis.feature


# plain text format

def parse(s, **kwargs):
    return [gws.gis.feature.new({
        'attributes': {'text': s}
    })]
