"""WFS ListStoredQueries template.

References:
    - https://schemas.opengis.net/wfs/2.0/examples/StoredQuery/ListStoredQueries.xml
    - https://mapserver.org/ogc/wfs_server.html#predefined-urn-ogc-def-query-ogc-wfs-getfeaturebyid-stored-query
"""

import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.plugin.ows_server.wfs
from gws.lib.xmlx import tag


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        tag('WFS:ListStoredQueriesResponse', doc(ta)),
        extra_namespaces=tpl.namespaces_from_caps(ta),
    )


def doc(ta):
    yield tag(
        'WFS:StoredQuery',
        {'id': gws.plugin.ows_server.wfs.STORED_QUERY_GET_FEATURE_BY_ID},
        tag('WFS:Title', 'Get Feature By Identifier'),
        [
            tag('WFS:ReturnFeatureType', tpl.feature_name(ta, lc))
            for lc in ta.layerCapsList
        ]
    )
