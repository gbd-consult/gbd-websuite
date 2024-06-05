import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.xmlx as xmlx
import gws.lib.date
import gws.gis.gml


def feature_collection(ta: server.TemplateArgs):
    return (
        'wfs:FeatureCollection', feature_collection_attributes(ta), [
            (
                f'wfs:member/{m.layerCaps.featureNameQ if m.layerCaps else "wfs:feature"}',
                {'gml:id': m.feature.uid()},
                feature_collection_member(ta, m)
            )
            for m in ta.featureCollection.members
        ]
    )


def value_collection(ta: server.TemplateArgs):
    return (
        'wfs:ValueCollection', feature_collection_attributes(ta), [
            (
                'wfs:member',
                format_value(ta, val)
            )
            for val in ta.featureCollection.values
        ]
    )


def feature_collection_attributes(ta):
    return {
        'timeStamp': ta.featureCollection.timestamp,
        'numberMatched': ta.featureCollection.numMatched,
        'numberReturned': ta.featureCollection.numReturned,
    }


def feature_collection_member(ta: server.TemplateArgs, m: server.FeatureCollectionMember):
    for name, val in sorted(m.feature.attributes.items()):
        if m.layerCaps:
            name = xmlx.namespace.qualify_name(name, m.layerCaps.xmlNamespace)
        yield name, format_value(ta, val)


def format_value(ta, val):
    if val is None:
        return ''
    if gws.lib.date.is_date_or_datetime(val):
        return val.isoformat()
    if isinstance(val, gws.Shape):
        # NB Qgis wants inline gml xmlns for adhoc schemas
        return gws.gis.gml.shape_to_element(
            val,
            version=ta.gmlVersion,
            always_xy=ta.sr.alwaysXY,
            with_inline_xmlns=True
        )
    return str(val)
