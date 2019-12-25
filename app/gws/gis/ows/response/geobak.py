import re

import gws.tools.xml3
import gws.gis.shape
import gws.gis.feature


# GeoBAK (https://www.egovernment.sachsen.de/geodaten.html)
#
# <geobak_20:Sachdatenabfrage...
#     <geobak_20:Kartenebene>....
#     <geobak_20:Inhalt>
#         <geobak_20:Datensatz>
#             <geobak_20:Attribut>
#                 <geobak_20:Name>...
#                 <geobak_20:Wert>...
#     <geobak_20:Inhalt>
#         <geobak_20:Datensatz>
#           ...
#

def parse(s, first_el, **kwargs):
    if 'geobak_20' not in first_el.namespaces:
        return None

    # some services have bare &'s in xml
    s = re.sub(r'&(?![#\w])', '', s)

    el = gws.tools.xml3.from_string(s)
    fs = []
    layer_name = el.get_text('Kartenebene')

    for content in el.all('Inhalt'):
        for ds in content.all('Datensatz'):
            atts = {
                a.get_text('Name').strip(): a.get_text('Wert').strip()
                for a in ds.all('Attribut')
            }
            fs.append(gws.gis.feature.new({
                'category': layer_name,
                'attributes': atts
            }))

    return fs
