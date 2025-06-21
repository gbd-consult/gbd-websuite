"""WFS ListStoredQueries template.

References:
    - https://schemas.opengis.net/wfs/2.0/examples/StoredQuery/ListStoredQueries.xml
    - https://mapserver.org/ogc/wfs_server.html#predefined-urn-ogc-def-query-ogc-wfs-getfeaturebyid-stored-query
"""

import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.plugin.ows_server.wfs


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        ('wfs:ListStoredQueriesResponse', doc(ta)),
        namespaces=tpl.namespaces_from_caps(ta),
    )


def doc(ta):
    yield (
        'wfs:StoredQuery',
        {'id': gws.plugin.ows_server.wfs.STORED_QUERY_GET_FEATURE_BY_ID},
        ('wfs:Title', 'Get Feature By Identifier'),
        [
            ('wfs:ReturnFeatureType', lc.featureNameQ)
            for lc in ta.layerCapsList
        ]
    )
