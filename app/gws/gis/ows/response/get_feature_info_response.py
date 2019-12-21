import gws.tools.xml3
import gws.gis.shape
import gws.gis.feature


# geoserver
#
# <GetFeatureInfoResponse>
#   <Layer name="....">
#       <Feature id="...">
#           <Attribute name="..." value="..."/>
#           <Attribute name="geometry" value="wkt"/>

def parse(s, first_el, **kwargs):
    if first_el.name.lower() != 'getfeatureinforesponse':
        return None

    el = gws.tools.xml3.from_string(s)
    fs = []

    for layer in el.all('Layer'):
        for feature in layer.all('Feature'):
            atts = {
                e.attr('name'): e.attr('value')
                for e in feature.all('Attribute')
            }

            shape = None
            if 'geometry' in atts:
                shape = gws.gis.shape.from_wkt(atts.pop('geometry'), kwargs.get('crs'))

            fs.append(gws.gis.feature.Feature({
                'uid': atts.get('uid') or feature.attr('id'),
                'category': layer.attr('name', ''),
                'shape': shape,
                'attributes': atts
            }))

    return fs
