"""WFS GetFeature template with GML 2."""

import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.xmlx as xmlx
import gws.plugin.ows_server.wfs.templates.common as common


def main(ta: server.TemplateArgs):
    ta.gmlVersion = 2
    return tpl.to_xml_response(
        ta,
        common.feature_collection(ta),
        extra_namespaces=[xmlx.namespace.require('gml2')]
    )
