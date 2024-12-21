"""CSW GetRecordByIdResponse template (ISO)."""

import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.plugin.ows_server.csw.templates.iso.record as rec


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        (
            'csw:GetRecordByIdResponse',
            {'version': ta.version},
            rec.record(ta, ta.metadataCollection.members[0])
        ),
    )
