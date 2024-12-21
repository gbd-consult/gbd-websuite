"""CSW GetRecords template (ISO)."""

import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.datetimex
import gws.lib.xmlx
import gws.plugin.ows_server.csw.templates.iso.record as rec


def main(ta: server.TemplateArgs):
    return tpl.to_xml_response(
        ta,
        ('csw:GetRecordsResponse', {'version': ta.version}, doc(ta)),
        extra_namespaces=[gws.lib.xmlx.namespace.get('gml')]
    )


def doc(ta: server.TemplateArgs):
    mdc = ta.metadataCollection

    yield 'csw:SearchStatus', {'timestamp': mdc.timestamp}
    yield (
        'csw:SearchResults', {
            'elementSet': 'full',
            'nextRecord': mdc.nextRecord,
            'numberOfRecordsMatched': mdc.numMatched,
            'numberOfRecordsReturned': mdc.numReturned,
            'recordSchema': 'http://www.isotc211.org/2005/gmd',
        }, [rec.record(ta, md) for md in mdc.members]
    )
