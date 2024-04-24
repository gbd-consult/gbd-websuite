import gws
import gws.base.ows.server as ows
import gws.base.ows.server.templatelib as tpl
import datetime
import gws.lib.uom
import gws.lib.xmlx as xmlx
import gws.gis.gml


def main(args: dict):
    ta = tpl.TemplateArgs(args)
    return tpl.to_xml(ta, ('wfs:FeatureCollection', feature_collection(ta)))


def feature_collection(ta: tpl.TemplateArgs):
    fc = ta.featureCollection

    yield {
        'timeStamp': fc.timestamp,
        'numberMatched': fc.numMatched,
        'numberReturned': fc.numReturned,
    }

    for m in fc.members:
        qname = xmlx.namespace.qualify_name(m.options.featureName, m.options.xmlNamespace)
        yield f'wfs:member/{qname}', member(ta, m)


def member(ta: tpl.TemplateArgs, m: ows.FeatureCollectionMember):
    yield {'gml:id': m.feature.uid()}

    for name, value in sorted(m.feature.attributes.items()):
        if name != m.feature.model.geometryName:
            qname = xmlx.namespace.qualify_name(name, m.options.xmlNamespace)
            yield qname, to_str(value)

    shape = m.feature.shape()
    if shape:
        el = gws.gis.gml.shape_to_element(shape, always_xy=ta.request.alwaysXY, with_inline_xmlns=True)
        qname = xmlx.namespace.qualify_name(m.options.geometryName, m.options.xmlNamespace)
        yield qname, el


def to_str(value):
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return str(value)
