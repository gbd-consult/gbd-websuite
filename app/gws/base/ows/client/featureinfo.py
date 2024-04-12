"""Parse WMS/WFS FeatureInfo responses."""

import gws
import gws.base.shape
import gws.gis.gml
import gws.lib.xmlx as xmlx
import gws.types as t


def parse(text: str, default_crs: gws.ICrs = None, always_xy=False) -> list[gws.FeatureRecord]:
    gws.debug.time_start('featureinfo:parse')
    res = _parse(text, default_crs, always_xy)
    gws.debug.time_end()
    return res


def _parse(text, default_crs, always_xy):
    try:
        xml_el = xmlx.from_string(text, case_insensitive=True, normalize_namespaces=True)
    except xmlx.Error:
        xml_el = None

    if xml_el:
        fn = _XML_FORMATS.get(xml_el.name)
        if fn:
            fds = fn(xml_el, default_crs, always_xy)
            if fds is not None:
                gws.log.debug(f'parsed with {fn.__name__} count={len(fds)}')
                return fds
        gws.log.error(f'XML parse error for {xml_el.name!r}')
        return

    # fallback for non-xml formats
    # @TODO: json etc

    for fn in _TEXT_FORMATS:
        fds = fn(text, default_crs, always_xy)
        if fds is not None:
            gws.log.debug(f'parsed with {fn.__name__} count={len(fds)}')
            return fds


##

def _parse_msgmloutput(xml_el: gws.IXmlElement, default_crs, always_xy):
    # msGMLOutput (MapServer)
    #
    # <msGMLOutput
    #     <LAYER_1>
    #         <gml:name>LAYER_NAME
    #         <FEATURE_1>
    #             <gml:boundedBy>
    #                    ...
    #             </gml:boundedBy>
    #             <GEOMETRY>
    #                 <gml:Point...
    #             </GEOMETRY>
    #             <attr>....</attr>
    #             <attr>....</attr>
    #

    fds = []

    for layer_el in xml_el:
        layer_name = layer_el.name
        for feature_el in layer_el:
            if _is_gml(feature_el) and feature_el.name == 'name':
                layer_name = feature_el.text
            else:
                fd = _fdata_from_gml(feature_el, default_crs, always_xy)
                fd.meta = {'layerName': layer_name}
                fds.append(fd)

    return fds


def _parse_featurecollection(xml_el: gws.IXmlElement, default_crs, always_xy):
    # FeatureCollection (OGC)
    #
    # <FeatureCollection
    #     <wfs:member>
    #         <FEATURE gml:id=...
    #             <attr>....</attr>
    #             <attr> <nested>....</attr>
    #             <GEOMETRY>
    #                 <gml:Point...
    #

    fds = []

    for member_el in xml_el:
        if member_el.name in {'member', 'featuremember'}:
            if len(member_el) == 1 and len(member_el[0]) > 0:
                # <wfs:member><my:feature><attr...
                fds.append(_fdata_from_gml(member_el[0], default_crs, always_xy))
            elif len(member_el) > 1:
                # <wfs:member><attr...
                fds.append(_fdata_from_gml(member_el, default_crs, always_xy))

    return fds


def _parse_getfeatureinforesponse(xml_el: gws.IXmlElement, default_crs, always_xy):
    # GetFeatureInfoResponse (geoserver/qgis)
    #
    # <GetFeatureInfoResponse>
    #      <Layer name="....">
    #          <Feature id="...">
    #              <Attribute name="..." value="..."/>
    #              <Attribute name="geometry" value="<wkt>"/>

    fds = []

    for layer_el in xml_el:
        layer_name = layer_el.get('name')
        for feature_el in layer_el:

            fd = gws.FeatureRecord(
                attributes={},
                uid=feature_el.get('id'),
                meta={'layerName': layer_name},
            )

            for el in feature_el:
                if el.name != 'attribute':
                    continue
                key = el.get('name')
                val = el.get('value')
                if key == 'geometry':
                    fd.shape = gws.base.shape.from_wkt(val, default_crs)
                elif val.strip():
                    fd.attributes[key] = val.strip()

            fds.append(fd)

    return fds


def _parse_featureinforesponse(xml_el: gws.IXmlElement, default_crs, always_xy):
    # FeatureInfoResponse (Arcgis)
    #
    # https://webhelp.esri.com/arcims/9.3/General/mergedProjects/wms_connect/wms_connector/get_featureinfo.htm
    #
    # <FeatureInfoResponse...
    #     <fields objectid="15111" shape="polygon"...
    #     <fields objectid="15111" shape="polygon"...

    fds = []

    for fields_el in xml_el:
        if fields_el.name == 'fields':
            fd = gws.FeatureRecord(attributes={})
            for key, val in fields_el.attrib.items():
                if key.lower() in {'id', 'fid'}:
                    fd.uid = val
                elif key.lower() != 'shape':
                    fd.attributes[key] = val
            fds.append(fd)

    return fds


def _parse_geobak(xml_el: gws.IXmlElement, default_crs, always_xy):
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

    fds = []

    for el in xml_el:
        fd = gws.FeatureRecord(attributes={})

        if el.name == 'kartenebene':
            fd.meta = {'layerName': el.text}
            continue

        if el.name == 'inhalt':
            for attr_el in el[0]:
                key = attr_el[0].text.strip()
                val = attr_el[1].text.strip()
                if key != 'shape' and val.lower() != 'null':
                    fd.attributes[key] = val

        fds.append(fd)

    return fds


##

_DEEP_ATTRIBUTE_DELIMITER = '.'


def _fdata_from_gml(feature_el, default_crs, always_xy) -> gws.FeatureRecord:
    # like GDAL does:
    # "When reading a feature, the driver will by default only take into account
    # the last recognized GML geometry found..." (https://gdal.org/drivers/vector/gml.html)

    fd = gws.FeatureRecord(
        attributes={},
        uid=feature_el.get('id') or feature_el.get('fid'),
        meta={'layerName': feature_el.name},
    )

    bbox = None

    for el in feature_el:
        if el.name == 'boundedby':
            # <gml:boundedBy directly under feature
            bbox = gws.gis.gml.parse_envelope(el[0], default_crs, always_xy)
        elif gws.gis.gml.is_geometry_element(el):
            # <gml:Polygon etc directly under feature
            fd.shape = gws.gis.gml.parse_shape(el, default_crs, always_xy)
        elif len(el) == 1 and gws.gis.gml.is_geometry_element(el[0]):
            # <gml:Polygon etc in a wrapper tag
            fd.shape = gws.gis.gml.parse_shape(el[0], default_crs, always_xy)
        elif len(el) > 0:
            # sub-feature
            sub = _fdata_from_gml(el, default_crs, always_xy)
            for k, v in sub.attributes.items():
                fd.attributes[el.name + _DEEP_ATTRIBUTE_DELIMITER + k] = v
        else:
            # attribute <attr>text</attr>
            s = el.text.strip()
            if s:
                fd.attributes[el.name] = s

    if not fd.shape and bbox:
        fd.shape = gws.base.shape.from_bounds(bbox)

    return fd


def _is_gml(el):
    return 'gml}' in el.tag


##


_XML_FORMATS = {
    'msgmloutput': _parse_msgmloutput,
    'featurecollection': _parse_featurecollection,
    'getfeatureinforesponse': _parse_getfeatureinforesponse,
    'featureinforesponse': _parse_featureinforesponse,
    'sachdatenabfrage': _parse_geobak,
}

_TEXT_FORMATS = [
]
