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
    return tpl.to_xml(
        ta,
        ('wfs:ListStoredQueriesResponse', doc(ta)),
        extra_namespaces=[lc.xmlNamespace for lc in ta.layerCapsList]
    )


def doc(ta):
    yield (
        'wfs:StoredQuery',
        {'id': gws.plugin.ows_server.wfs.STORED_QUERY_GET_FEATURE_BY_ID},
        ('wfs:Title', 'Get Feature By Identifier'),
        [
            ('wfs:ReturnFeatureType', lc.featureQname)
            for lc in ta.layerCapsList
        ]
    )
