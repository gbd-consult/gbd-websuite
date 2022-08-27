"""Parse WMS/WFS FeatureInfo responses."""

import gws
import gws.base.shape
import gws.gis.gml
import gws.lib.xml2 as xml2
import gws.types as t


def parse(
    text,
    fallback_crs: t.Optional[gws.ICrs] = None,
    **kwargs
) -> t.List[gws.IFeature]:
    ts = gws.time_start('featureinfo:parse')
    res = _parse(text, fallback_crs, **kwargs)
    gws.time_end(ts)
    return res


def _parse(text, fallback_crs, **kwargs):
    try:
        xml_el = xml2.from_string(text, strip_ns=True)
    except xml2.Error:
        xml_el = None

    if xml_el:
        for fn in _XML_FORMATS:
            features = fn(xml_el, fallback_crs, **kwargs)
            if features is not None:
                gws.log.debug(f'parsed with {fn.__name__} count={len(features)}')
                return features

    # fallback for non-xml formats
    for fn in _TEXT_FORMATS:
        features = fn(text, fallback_crs, **kwargs)
        if features is not None:
            gws.log.debug(f'parsed with {fn.__name__} count={len(features)}')
            return features


##

_DEEP_PROP_NAME_DELIMITER = '/'


def _f_FeatureCollection(el: gws.XmlElement, fallback_crs, **kwargs):
    # wfs FeatureCollection
    #
    # <FeatureCollection...
    #     <featureMember>...
    #         <...>

    if not xml2.element_is(el, 'FeatureCollection'):
        return None

    features = []

    for member_el in xml2.all(el, 'member', 'featureMember'):
        if not member_el.children:
            continue

        content_el = member_el.children[0] if len(member_el.children) == 1 else member_el

        atts = []
        shapes = []
        uid = xml2.attr(content_el, 'id', 'fid')

        # flatten a potentially deeply nested structure into a flat list,
        # with property names = tags joined by the delim

        for c in content_el.children:
            _fc_walk(c, '', atts, shapes, fallback_crs)

        # like GDAL does:
        # "When reading a feature, the driver will by default only take into account
        # the last recognized GML geometry found..." (https://gdal.org/drivers/vector/gml.html)

        shape = shapes[-1] if shapes else None

        features.append(gws.base.feature.from_args(
            uid=uid,
            category=content_el.name,
            shape=shape,
            attributes=atts
        ))

    return features


def _fc_walk(el: gws.XmlElement, path, atts, shapes, fallback_crs):
    path = (path + _DEEP_PROP_NAME_DELIMITER + el.name) if path else el.name

    if not el.children and el.text:
        atts.append(gws.Attribute(name=path, value=el.text))
        return

    if gws.gis.gml.element_is_gml(el):
        shapes.append(gws.gis.gml.parse_to_shape(el, fallback_crs=fallback_crs))
        return

    if xml2.element_is(el, 'boundedBy'):
        return

    for c in el.children:
        _fc_walk(c, path, atts, shapes, fallback_crs)


##

def _f_GetFeatureInfoResponse(el: gws.XmlElement, fallback_crs, **kwargs):
    # geoserver/qgis
    #
    # <GetFeatureInfoResponse>
    #      <Layer name="....">
    #          <Feature id="...">
    #              <Attribute name="..." value="..."/>
    #              <Attribute name="geometry" value="<wkt>"/>

    if not xml2.element_is(el, 'GetFeatureInfoResponse'):
        return None

    features = []

    for layer_el in xml2.all(el, 'Layer'):

        layer_name = xml2.attr(layer_el, 'name')

        for feature_el in xml2.all(layer_el, 'Feature'):

            uid = xml2.attr(feature_el, 'id', 'fid')
            atts = []
            shape = None

            for attr_el in xml2.all(feature_el, 'Attribute'):
                name = xml2.attr(attr_el, 'name')
                value = xml2.attr(attr_el, 'value')

                if value == 'null':
                    continue

                if name.lower() == 'geometry':
                    try:
                        shape = gws.base.shape.from_wkt(value, fallback_crs)
                    except Exception:
                        gws.log.exception()
                        continue

                if name.lower() in {'id', 'uid', 'fid'}:
                    uid = value
                    continue

                atts.append(gws.Attribute(name=name, value=value))

            features.append(gws.base.feature.from_args(
                uid=uid,
                category=layer_name or '',
                shape=shape,
                attributes=atts
            ))

    return features


##

def _f_FeatureInfoResponse(el: gws.XmlElement, fallback_crs, **kwargs):
    # esri
    #
    # <FeatureInfoResponse...
    #     <fields objectid="15111" shape="polygon"...
    #     <fields objectid="15111" shape="polygon"...

    if not xml2.element_is(el, 'GetFeatureInfoResponse'):
        return None

    features = []

    for field_el in xml2.all(el, 'Fields'):
        atts = []
        uid = ''

        for k, v in field_el.attributes:
            if k.lower() == 'objectid':
                uid = v
            else:
                atts.append(gws.Attribute(name=k, value=v))

        features.append(gws.base.feature.from_args(
            uid=uid,
            attributes=atts
        ))

    return features


##

def _f_GeoBAK(el: gws.XmlElement, fallback_crs, **kwargs):
    # GeoBAK (https://www.egovernment.sachsen.de/geodaten.html)
    #
    # <geobak_20:Sachdatenabfrage...
    #     <geobak_20:Kartenebene>....
    #     <geobak_20:Inhalt>
    #         <geobak_20:Datensatz>
    #             <geobak_20:Attribut>
    #                 <geobak_20:Name>...
    #                 <geobak_20:Wert>...
    #     <geobak_20:Inhalt>
    #         <geobak_20:Datensatz>
    #           ...
    #

    if not xml2.element_is(el, 'Sachdatenabfrage'):
        return None

    features = []

    layer_name = xml2.text(el, 'Kartenebene')

    for content_el in xml2.all(el, 'Inhalt'):
        for feature_el in xml2.all(content_el, 'Datensatz'):
            atts = {
                xml2.text(a, 'Name').strip(): xml2.text(a, 'Wert').strip()
                for a in xml2.all(feature_el, 'Attribut')
            }
            features.append(gws.base.feature.from_args(
                category=layer_name,
                attributes=atts
            ))

    return features


##


_XML_FORMATS = [
    _f_FeatureCollection,
    _f_GetFeatureInfoResponse,
    _f_FeatureInfoResponse,
    _f_GeoBAK,
]

_TEXT_FORMATS = [
]
