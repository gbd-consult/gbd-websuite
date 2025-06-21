"""WFS GetPropertyValue template with GML 3."""

import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl


def main(ta: server.TemplateArgs):
    ta.gmlVersion = 3
    return tpl.to_xml_response(
        ta,
        tpl.wfs_value_collection(ta),
    )
