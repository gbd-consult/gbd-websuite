import gws.tools.xml3
import gws.gis.shape
import gws.gis.feature


# ESRI
#
# <FeatureInfoResponse...
#   <fields objectid="15111" shape="polygon"...
#   <fields objectid="15111" shape="polygon"...

def parse(s, first_el, **kwargs):
    if first_el.name.lower() != 'featureinforesponse':
        return None

    el = gws.tools.xml3.from_string(s)
    fs = []

    for item in el.all('Fields'):
        fs.append(gws.gis.feature.Feature({
            'attributes': item.attr_dict
        }))

    return fs
