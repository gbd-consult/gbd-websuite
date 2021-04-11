import gws.tools.xml2
import gws.gis.shape
import gws.gis.feature


# geoserver
#
# <GetFeatureInfoResponse>
#   <Layer name="....">
#       <Feature id="...">
#           <Attribute name="..." value="..."/>
#           <Attribute name="geometry" value="wkt"/>

def parse(text, first_el, crs=None, invert_axis=None, **kwargs):
    if first_el.name.lower() != 'getfeatureinforesponse':
        return None

    el = gws.tools.xml2.from_string(text)
    fs = []

    for layer in el.all('Layer'):
        for feature in layer.all('Feature'):
            atts = {}

            for e in feature.all('Attribute'):
                name = e.attr('name')
                value = e.attr('value')
                if gws.as_str(value).lower() != 'null':
                    atts[name] = value

            shape = None
            if 'geometry' in atts:
                shape = gws.gis.shape.from_wkt(atts.pop('geometry'), crs)

            fs.append(gws.gis.feature.Feature(
                uid=atts.get('uid') or feature.attr('id'),
                category=layer.attr('name', ''),
                shape=shape,
                attributes=atts
            ))

    return fs
