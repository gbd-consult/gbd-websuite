"""WFS DescribeStoredQueries template.

References:
    - https://schemas.opengis.net/wfs/2.0/examples/StoredQuery/DescribeStoredQueries.xml
    - https://mapserver.org/ogc/wfs_server.html#predefined-urn-ogc-def-query-ogc-wfs-getfeaturebyid-stored-query
"""

import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.xmlx as xmlx
import gws.plugin.ows_server.wfs


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        ('wfs:DescribeStoredQueriesResponse', doc(ta)),
        namespaces={
            'xsd': xmlx.namespace.require('xsd'),
            **tpl.namespaces_from_caps(ta),
        }
    )


def doc(ta):
    types = ' '.join(lc.featureNameQ for lc in ta.layerCapsList)
    yield (
        'wfs:StoredQueryDescription',
        {'id': gws.plugin.ows_server.wfs.STORED_QUERY_GET_FEATURE_BY_ID},
        ('wfs:Title', 'Get Feature By Identifier'),
        ('wfs:Abstract', 'This stored query will retrieve the feature whose identifier matches the one specified as the argument to this method.'),
        ('wfs:Parameter', {'name': 'id', 'type': 'xsd:string'}),
        (
            'wfs:QueryExpressionText',
            {
                'isPrivate': 'true',
                'language': 'urn:ogc:def:queryLanguage:OGC-WFS::WFS_QueryExpression',
                'returnFeatureTypes': types,
            },
            (
                'wfs:Query',
                {'typeNames': types},
                ('fes:Filter/fes:ResourceId', {'rid': '${ID}'}),
            ),
        ),
    )
