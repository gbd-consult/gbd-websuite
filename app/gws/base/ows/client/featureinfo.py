"""Parse WMS/WFS FeatureInfo responses."""

import gws
import gws.base.shape
import gws.lib.gml
import gws.lib.xmlx as xmlx


class Error(gws.Error):
    pass


def parse(text: str, default_crs: gws.Crs = None, always_xy=False) -> list[gws.FeatureRecord]:
    gws.debug.time_start('featureinfo:parse')
    recs = _parse(text.strip(), default_crs, always_xy)
    gws.debug.time_end()
    return recs


def _parse(text, default_crs, always_xy):
    if not text.strip():
        return []

    if text.startswith('<'):
        try:
            xml_el = xmlx.from_string(text, gws.XmlOptions(removeNamespaces=True))
        except xmlx.Error as exc:
            raise Error(f'XML error') from exc

        parser = _XML_FORMATS.get(xml_el.lcName)
        if not parser:
            raise Error(f'XML format error for {xml_el.name!r}')

        recs = parser(xml_el, default_crs, always_xy)
        gws.log.debug(f'parsed with {parser.__name__} count={len(recs)}')
        return recs

    raise Error(f'unknown format in {text[:100]!r}')


##


def _parse_msgmloutput(xml_el: gws.XmlElement, default_crs, always_xy):
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

    recs = []

    for layer_el in xml_el:
        layer_name = layer_el.lcName
        for el in layer_el:
            if el.lcName == 'name':
                layer_name = el.text
            else:
                rec = _record_from_gml(el, default_crs, always_xy)
                rec.meta = {'layerName': layer_name}
                recs.append(rec)

    return recs


def _parse_featurecollection(xml_el: gws.XmlElement, default_crs, always_xy):
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

    recs = []

    for member_el in xml_el:
        if member_el.lcName in {'member', 'featuremember'}:
            if len(member_el) == 1 and len(member_el[0]) > 0:
                # <wfs:member><my:feature><attr...
                recs.append(_record_from_gml(member_el[0], default_crs, always_xy))
            elif len(member_el) > 1:
                # <wfs:member><attr...
                recs.append(_record_from_gml(member_el, default_crs, always_xy))

    return recs


def _parse_getfeatureinforesponse(xml_el: gws.XmlElement, default_crs, always_xy):
    # GetFeatureInfoResponse (geoserver/qgis)
    #
    # <GetFeatureInfoResponse>
    #      <Layer name="....">
    #          <Feature id="...">
    #              <Attribute name="..." value="..."/>
    #              <Attribute name="geometry" value="<wkt>"/>
    #
    # For qgis raster layers, "Attribute" is directly under "Layer":
    #
    # <GetFeatureInfoResponse>
    #      <Layer name="....">
    #              <Attribute name="..." value="..."/>

    def attr(rec, el):
        key = el.get('name').lower()
        val = el.get('value', '').strip()        
        if key == 'geometry':
            rec.shape = gws.base.shape.from_wkt(val, default_crs)
        elif len(val) > 0:
            rec.attributes[key] = val

    recs = []

    for layer_el in xml_el:
        layer_name = layer_el.get('name')

        raster_rec = gws.FeatureRecord(
            attributes={},
            uid='',
            meta={'layerName': layer_name},
        )

        for sub_el in layer_el:
            if sub_el.lcName == 'feature':
                rec = gws.FeatureRecord(
                    attributes={},
                    uid=_get_uid(sub_el),
                    meta={'layerName': layer_name},
                )
                for el in sub_el:
                    if el.lcName == 'attribute':
                        attr(rec, el)
                recs.append(rec)
            
            if sub_el.lcName == 'attribute':
                attr(raster_rec, sub_el)

        if raster_rec.attributes:
            recs.append(raster_rec)

    return recs


def _parse_featureinforesponse(xml_el: gws.XmlElement, default_crs, always_xy):
    # FeatureInfoResponse (Arcgis)
    #
    # https://webhelp.esri.com/arcims/9.3/General/mergedProjects/wms_connect/wms_connector/get_featureinfo.htm
    #
    # <FeatureInfoResponse...
    #     <fields objectid="15111" shape="polygon"...
    #     <fields objectid="15111" shape="polygon"...

    recs = []

    for fields_el in xml_el:
        if fields_el.lcName == 'fields':
            rec = gws.FeatureRecord(
                attributes={},
                uid=_get_uid(fields_el),
            )
            for key, val in fields_el.attrib.items():
                key = key.lower()
                if key != 'shape':
                    rec.attributes[key] = val
            recs.append(rec)

    return recs


def _parse_geobak(xml_el: gws.XmlElement, default_crs, always_xy):
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

    recs = []

    for el in xml_el:
        rec = gws.FeatureRecord(attributes={})

        if el.lcName == 'kartenebene':
            rec.meta = {'layerName': el.text}
            continue

        if el.lcName == 'inhalt':
            for attr_el in el[0]:
                key = attr_el[0].text.strip().lower()
                val = attr_el[1].text.strip()
                if key != 'shape' and val.lower() != 'null':
                    rec.attributes[key] = val

        recs.append(rec)

    return recs


##

_DEEP_ATTRIBUTE_DELIMITER = '.'


def _record_from_gml(feature_el, default_crs, always_xy) -> gws.FeatureRecord:
    # like GDAL does:
    # "When reading a feature, the driver will by default only take into account
    # the last recognized GML geometry found..." (https://gdal.org/drivers/vector/gml.html)

    rec = gws.FeatureRecord(
        attributes={},
        uid=_get_uid(feature_el),
        meta={'layerName': feature_el.lcName},
    )

    bbox = None

    for el in feature_el:
        if el.lcName == 'boundedby':
            # <gml:boundedBy directly under feature
            bbox = gws.lib.gml.parse_envelope(el[0], default_crs, always_xy)
        elif gws.lib.gml.is_geometry_element(el):
            # <gml:Polygon etc directly under feature
            rec.shape = gws.lib.gml.parse_shape(el, default_crs, always_xy)
        elif len(el) == 1 and gws.lib.gml.is_geometry_element(el[0]):
            # <gml:Polygon etc in a wrapper tag
            rec.shape = gws.lib.gml.parse_shape(el[0], default_crs, always_xy)
        elif len(el) > 0:
            # sub-feature
            sub = _record_from_gml(el, default_crs, always_xy)
            for k, v in sub.attributes.items():
                rec.attributes[el.lcName + _DEEP_ATTRIBUTE_DELIMITER + k] = v
        else:
            # attribute <attr>text</attr>
            s = el.text.strip()
            if s:
                rec.attributes[el.lcName] = s

    if not rec.shape and bbox:
        rec.shape = gws.base.shape.from_bounds(bbox)

    return rec


_UIDS = ['id', 'fid', 'objectid', 'ID', 'FID', 'OBJECTID']


def _get_uid(el):
    for u in _UIDS:
        if u in el.attrib:
            return el.get(u)
    return ''


##


_XML_FORMATS = {
    'msgmloutput': _parse_msgmloutput,
    'featurecollection': _parse_featurecollection,
    'getfeatureinforesponse': _parse_getfeatureinforesponse,
    'featureinforesponse': _parse_featureinforesponse,
    'sachdatenabfrage': _parse_geobak,
}
