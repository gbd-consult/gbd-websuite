"""WFS GetPropertyValue template with GML 3."""

import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.xmlx as xmlx
import gws.plugin.ows_server.wfs.templates.common as common


def main(ta: server.TemplateArgs):
    ta.gmlVersion = 3
    return tpl.to_xml_response(
        ta,
        common.value_collection(ta),
        extra_namespaces=[xmlx.namespace.get('gml3')]
    )
