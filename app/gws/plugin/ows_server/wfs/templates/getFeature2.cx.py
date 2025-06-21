"""WFS GetFeature template with GML 2."""

import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.xmlx as xmlx


def main(ta: server.TemplateArgs):
    ta.gmlVersion = 2
    return tpl.to_xml_response(
        ta,
        tpl.wfs_feature_collection(ta),
        namespaces={
            'gml': xmlx.namespace.require('gml2'),
        },
    )
