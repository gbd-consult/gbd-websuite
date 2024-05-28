"""GetFeature template with GML 3."""

import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.xmlx as xmlx


def main(ta: server.TemplateArgs):
    ta.gmlVersion = 3
    return tpl.to_xml(
        ta,
        tpl.wfs_feature_collection(ta),
        extra_namespaces=[xmlx.namespace.get('gml3')]
    )
