import gws.lib.feature
import gws.lib.xml2

# ESRI
#
# <FeatureInfoResponse...
#   <fields objectid="15111" shape="polygon"...
#   <fields objectid="15111" shape="polygon"...

def parse(text, first_el, crs=None, invert_axis=None, **kwargs):
    if first_el.name.lower() != 'featureinforesponse':
        return None

    el = gws.lib.xml2.from_string(text)
    fs = []

    for item in el.all('Fields'):
        atts = item.attr_dict
        uid = atts.pop('objectid', None) or atts.pop('OBJECTID', None)
        fs.append(gws.lib.feature.Feature(uid=uid, attributes=atts))

    return fs
