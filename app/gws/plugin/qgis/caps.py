"""QGIS project xml parser."""

from typing import Optional

import math
import re

import gws
import gws.gis.bounds
import gws.gis.extent
import gws.gis.crs
import gws.gis.source
import gws.lib.metadata
import gws.lib.net
import gws.lib.xmlx


class PrintTemplateElement(gws.Data):
    type: str
    uuid: str
    attributes: dict
    position: gws.UomPoint
    size: gws.UomSize


class PrintTemplate(gws.Data):
    title: str
    index: int
    attributes: dict
    elements: list[PrintTemplateElement]


class Caps(gws.Data):
    metadata: gws.Metadata
    printTemplates: list[PrintTemplate]
    projectCrs: gws.Crs
    projectBounds: Optional[gws.Bounds]
    properties: dict
    sourceLayers: list[gws.SourceLayer]
    version: str
    visibilityPresets: dict[str, list[str]]


def parse(xml: str) -> Caps:
    el = gws.lib.xmlx.from_string(xml)
    return parse_element(el)


def parse_element(root_el: gws.XmlElement) -> Caps:
    caps = Caps()

    caps.version = root_el.get('version')
    caps.properties = parse_properties(root_el.find('properties'))
    caps.metadata = _project_metadata(root_el)
    caps.printTemplates = _print_templates(root_el)
    caps.visibilityPresets = _visibility_presets(root_el)

    srid = root_el.textof('projectCrs/spatialrefsys/authid') or '4326'
    caps.projectCrs = gws.gis.crs.get(srid)
    if not caps.projectCrs:
        raise gws.Error(f'invalid CRS in qgis project')
    caps.projectBounds = _project_bounds(root_el, caps.projectCrs)

    layers_dct = _map_layers(root_el, caps)
    root_group = _layer_tree(root_el.find('layer-tree-group'), layers_dct)
    caps.sourceLayers = gws.gis.source.check_layers(root_group.layers)

    return caps


##


def _project_metadata(root_el) -> gws.Metadata:
    md = gws.Metadata()

    el = root_el.find('projectMetadata')
    if el:
        _metadata(el, md)

    # @TODO supplementary metadata
    return md


def _project_bounds(root_el, project_crs: gws.Crs):
    # the project extent can be defined in Project->QGIS Server->Advertised extent
    #     <properties>
    #         <WMSExtent...
    #             <value>...</value>
    #             <value>...</value>
    #             <value>...</value>
    #             <value>...</value>
    #         </WMSExtent>

    # NB not using map canvas extent because it's volatile and can be changed accidentally

    ext = _extent_from_tag(root_el.find('properties/WMSExtent'))
    if ext:
        return gws.gis.bounds.from_extent(ext, project_crs)


_meta_mapping = [
    ('authorityIdentifier', 'identifier'),
    ('parentIdentifier', 'parentidentifier'),
    ('language', 'language'),
    ('type', 'type'),
    ('title', 'title'),
    ('abstract', 'abstract'),
    ('dateCreated', 'creation'),
    ('fees', 'fees'),
]

_contact_mapping = [
    ('contactEmail', 'email'),
    ('contactFax', 'fax'),
    ('contactOrganization', 'organization'),
    ('contactPerson', 'name'),
    ('contactPhone', 'voice'),
    ('contactPosition', 'position'),
    ('contactRole', 'role'),
    ('contactAddress', 'address'),
    ('contactAddressType', 'type'),
    ('contactArea', 'administrativearea'),
    ('contactCity', 'city'),
    ('contactCountry', 'country'),
    ('contactZip', 'postalcode'),
]


def _add_dict(dst, src, mapping):
    for dkey, skey in mapping:
        if skey in src:
            setattr(dst, dkey, src[skey])


def _metadata(el: gws.XmlElement, md: gws.Metadata):
    # extract metadata from projectMetadata/resourceMetadata

    _add_dict(md, el.textdict(), _meta_mapping)

    md.keywords = []
    for kw in el.findall('keywords'):
        keywords = kw.textlist('keyword')
        if kw.get('vocabulary') == 'gmd:topicCategory':
            md.isoTopicCategories = keywords
        else:
            md.keywords.extend(keywords)

    contact_el = el.find('contact')
    if contact_el:
        _add_dict(md, contact_el.textdict(), _contact_mapping)
        for e in contact_el.findall('contactAddress'):
            _add_dict(md, e.textdict(), _contact_mapping)
            break  # NB we only support one contact address

    md.metaLinks = []
    for e in el.findall('links/link'):
        # @TODO clarify
        md.metaLinks.append(gws.MetadataLink(
            url=e.get('url'),
            description=e.get('description'),
            mimeType=e.get('mimeType'),
            format=e.get('format'),
            title=e.get('name'),
            scheme=e.get('type'),
        ))

    md.accessConstraints = []
    for e in el.findall('constraints'):
        md.accessConstraints.append(gws.MetadataAccessConstraint(
            type=e.get('type'),
            text=e.text,
        ))

    for e in el.findall('license'):
        md.license = gws.MetadataLicense(name=e.text)
        break

    e = el.find('extent/temporal')
    if e:
        md.dateBegin = e.textof('period/start')
        md.dateEnd = e.textof('period/end')


# see QGIS/src/core/layout/qgslayoutitemregistry.h

_QGraphicsItem_UserType = 65536  # https://doc.qt.io/qtforpython/PySide2/QtWidgets/QGraphicsItem.html

_LT0 = _QGraphicsItem_UserType + 100

_LAYOUT_TYPES = {
    _LT0 + 0: 'item',  # LayoutItem
    _LT0 + 1: 'group',  # LayoutGroup
    _LT0 + 2: 'page',  # LayoutPage
    _LT0 + 3: 'map',  # LayoutMap
    _LT0 + 4: 'picture',  # LayoutPicture
    _LT0 + 5: 'label',  # LayoutLabel
    _LT0 + 6: 'legend',  # LayoutLegend
    _LT0 + 7: 'shape',  # LayoutShape
    _LT0 + 8: 'polygon',  # LayoutPolygon
    _LT0 + 9: 'polyline',  # LayoutPolyline
    _LT0 + 10: 'scalebar',  # LayoutScaleBar
    _LT0 + 11: 'frame',  # LayoutFrame
    _LT0 + 12: 'html',  # LayoutHtml
    _LT0 + 13: 'attributetable',  # LayoutAttributeTable
    _LT0 + 14: 'texttable',  # LayoutTextTable
    _LT0 + 15: '3dmap',  # Layout3DMap
    _LT0 + 16: 'manualtable',  # LayoutManualTable
    _LT0 + 17: 'marker',  # LayoutMarker
}


# print templates in qgis-3:
#
# <Layouts>
#    <Layout name="..."  <- template 1
#       <PageCollection
#           <LayoutItem <- pages
#       <LayoutItem type="<int, see below>" ...
#       <LayoutMultiFrame type="<int>" ...
#       <Layout??? <- evtl. other item tags
#
#    <Layout name="..." <- template 2
#      etc


def _print_templates(root_el: gws.XmlElement):
    templates = []

    for layout_el in root_el.findall('Layouts/Layout'):
        tpl = PrintTemplate(
            title=layout_el.get('name', ''),
            attributes=layout_el.attrib,
            index=len(templates),
            elements=[],
        )

        pc_el = layout_el.find('PageCollection')
        if pc_el:
            tpl.elements.extend(gws.u.compact(_layout_element(c) for c in pc_el))

        tpl.elements.extend(gws.u.compact(_layout_element(c) for c in layout_el))

        templates.append(tpl)

    return templates


def _layout_element(item_el: gws.XmlElement):
    type = _LAYOUT_TYPES.get(_parse_int(item_el.get('type')))
    uuid = item_el.get('uuid')
    if type and uuid:
        return PrintTemplateElement(
            type=type,
            uuid=uuid,
            attributes=item_el.attrib,
            position=_parse_msize(item_el.get('position')),
            size=_parse_msize(item_el.get('size')),
        )


##


def _map_layers(root_el: gws.XmlElement, caps: Caps) -> dict[str, gws.SourceLayer]:
    no_wms_layers = set(caps.properties.get('WMSRestrictedLayers', []))
    use_layer_ids = caps.properties.get('WMSUseLayerIDs', False)

    map_layers = {}

    for el in root_el.findall('projectlayers/maplayer'):
        sl = _map_layer(el)
        if not sl:
            continue

        title = sl.metadata.get('title')

        # no_wms_layers always contains titles, not ids (=names)
        if title in no_wms_layers:
            continue

        uid = el.textof('id')
        if use_layer_ids:
            name = uid
        else:
            name = el.textof('shortname') or el.textof('layername')

        sl.title = title
        sl.name = name
        sl.sourceId = uid

        ext = _map_layer_wgs_extent(el, caps.projectCrs)
        if ext:
            sl.wgsExtent = ext

        map_layers[uid] = sl

    return map_layers


def _map_layer(layer_el: gws.XmlElement):
    sl = gws.SourceLayer(
        supportedCrs=[],
    )

    sl.metadata = _map_layer_metadata(layer_el)

    crs = gws.gis.crs.get(layer_el.textof('srs/spatialrefsys/authid'))
    if crs:
        sl.supportedCrs.append(crs)

    if layer_el.get('hasScaleBasedVisibilityFlag') == '1':
        # in qgis, maxScale < minScale
        a = _parse_float(layer_el.get('maxScale'))
        z = _parse_float(layer_el.get('minScale'))
        if z > a:
            sl.scaleRange = [a, z]

    sl.dataSource = _layer_datasource(layer_el)

    sl.opacity = _parse_float(layer_el.textof('layerOpacity') or '1')
    sl.isQueryable = layer_el.textof('flags/Identifiable') == '1'

    sl.properties = parse_properties(layer_el.find('customproperties'))

    return sl


def _map_layer_metadata(layer_el) -> gws.Metadata:
    # Layer metadata is either Layer->Properties->Metadata (stored in maplayer/resourceMetadata),
    # or Layer->Properties->QGIS Server (stored directly under maplayer/abstract, maplayer/keywordList and so on.

    md = gws.Metadata()

    el = layer_el.find('resourceMetadata')
    if el:
        _metadata(el, md)

    # fill in missing props from direct metadata

    d = layer_el.textdict()

    md.title = md.title or d.get('title') or d.get('shortname') or d.get('layername')
    md.abstract = md.abstract or d.get('abstract')

    if not md.attribution and d.get('attribution'):
        md.attribution = gws.MetadataAttribution(title=d.get('attribution'))

    if not md.keywords:
        md.keywords = layer_el.textlist('keywordList/value')

    if not md.metaLinks:
        md.metaLinks = []
        for e in layer_el.findall('metadataUrls/metadataUrl'):
            md.metaLinks.append(gws.MetadataLink(
                type=e.get('type'),
                format=e.get('format'),
                title=e.text,
            ))

    return md


def _map_layer_wgs_extent(layer_el: gws.XmlElement, project_crs: gws.Crs):
    # extent explicitly defined in metadata (Layer Props -> Metadata -> Extent)
    el = layer_el.find('resourceMetadata/extent/spatial')
    if el:
        ext = _extent_from_atts(el)
        crs = gws.gis.crs.get(el.get('crs'))
        if ext and crs:
            ext = gws.gis.extent.transform(ext, crs, gws.gis.crs.WGS84)
            if gws.gis.extent.is_valid(ext):
                gws.log.debug(f"_map_layer_wgs_extent: {layer_el.textof('id')}: spatial: {ext}")
                return ext

    # extent in <maplayer>/<wgs84extent>
    ext = _extent_from_tag(layer_el.find('wgs84extent'))
    if gws.gis.extent.is_valid(ext):
        gws.log.debug(f"_map_layer_wgs_extent: {layer_el.textof('id')}: wgs84extent: {ext}")
        return ext

    # extent in <maplayer>/<extent>, assume the project CRS
    ext = _extent_from_tag(layer_el.find('extent'))
    if gws.gis.extent.is_valid(ext):
        gws.log.debug(f"_map_layer_wgs_extent: {layer_el.textof('id')}: extent: {ext}")
        return gws.gis.bounds.wgs_extent(gws.Bounds(extent=ext, crs=project_crs))

    gws.log.warning(f"_map_layer_wgs_extent: {layer_el.textof('id')}: NOT FOUND")


# layer trees:

# <layer-tree-group>
#     <layer-tree-group checked="Qt::Checked" expanded="1" name="...">
#         <layer-tree-layer ... checked="Qt::Checked" expanded="1" id="...">
#         ...


def _layer_tree(el: gws.XmlElement, layers_dct):
    visible = el.get('checked') != 'Qt::Unchecked'
    expanded = el.get('expanded') == '1'

    if el.tag == 'layer-tree-group':
        title = el.get('name')
        # qgis doesn't write 'id' for groups but our generators might
        name = el.get('id') or title

        return gws.SourceLayer(
            title=title,
            name=name,
            metadata=gws.Metadata(title=title, name=name),
            isVisible=visible,
            isExpanded=expanded,
            isGroup=True,
            isQueryable=False,
            isImage=False,
            layers=gws.u.compact(_layer_tree(c, layers_dct) for c in el)
        )

    if el.tag == 'layer-tree-layer':
        sl = layers_dct.get(el.get('id'))
        if sl:
            sl.isVisible = visible
            sl.isExpanded = expanded
            sl.isGroup = False
            sl.isImage = True
            return sl


def _layer_datasource(layer_el: gws.XmlElement) -> dict:
    prov = layer_el.textof('provider')
    ds_text = layer_el.textof('datasource')

    if ds_text:
        return parse_datasource((prov or '').lower(), ds_text)
    if prov:
        return {'provider': prov.lower()}
    return {}


##

def _visibility_presets(root_el: gws.XmlElement):
    """Parse the global ``visibility-presets`` block.

    We're only interested in which layers are visible.

    Overall structure::

        <visibility-presets>
            <visibility-preset .... name="...">
                <layer id="..." visible="1" ... />
                <layer id="..." visible="1" ... />
            <visibility-preset .... name="...">
                <layer id="..." visible="1" ... />
                <layer id="..." visible="1" ... />

    """

    d = {}

    for el in root_el.findall('visibility-presets/visibility-preset'):
        ls = []
        for la in el.findall('layer'):
            if la.attr('visible') == '1':
                ls.append(la.attr('id'))
        d[el.attr('name')] = ls

    return d


##


def parse_datasource(prov, text):
    ds = gws.u.to_lower_dict(_parse_datasource(text) or {})
    ds['provider'] = (ds.get('provider') or prov).lower()

    if ds['provider'] == 'wms' and 'tilematrixset' in ds:
        ds['provider'] = 'wmts'
    elif ds['provider'] == 'wms' and ds.get('type') == 'xyz':
        ds['provider'] = 'xyz'

    # @TODO classify ogr's based on a file extension

    return ds


def _parse_datasource(text):
    # Datasources are very versatile and the format depends on the provider.
    # For some hints see `decodedSource` in qgsvectorlayer.cpp/qgsrasterlayer.cpp.
    # We don't have ambition to parse them all, just do some ad-hoc parsing
    # of the most common flavors, and return the rest as `{'text': text}`.

    text = text.strip()

    if re.match(r'^\w+=[^&]*&', text):
        # key=value, amp-separated, uri-encoded
        # used for WMS, e.g.
        # contextualWMSLegend=0&crs=EPSG:31468&...&url=...?SERVICE%3DWMTS%26REQUEST%3DGetCapabilities
        return _datasource_amp_delimited(text)

    if re.match(r'^\w+=\S+ ', text):
        # key=value, space separated
        # used for postgres/WFS, e.g.
        # dbname='...' host=... port=...
        # pagingEnabled='...' preferCoordinatesForWfsT11=...
        return _datasource_space_delimited(text)

    if text.startswith(('http://', 'https://')):
        # just an url
        return {'url': text}

    if text.startswith(('.', '/')):
        # path or path|options
        # used for Geojson, GPKG, e.g.
        # ../rel/path/test.gpkg|layername=name|subset=... etc
        return _datasource_pipe_delimited(text)

    return {'text': text}


def _datasource_amp_delimited(text):
    ds = {}

    for p in text.split('&'):
        if '=' not in p:
            continue
        k, v = p.split('=', maxsplit=1)

        v = gws.lib.net.unquote(v)

        if k in {'layers', 'styles'}:
            ds.setdefault(k, []).append(v)
        else:
            ds[k] = v

    if 'url' not in ds:
        return ds

    # extract params from the url

    url, params = gws.lib.net.extract_params(ds['url'])

    if 'typename' in params:
        ds['typename'] = params.pop('typename')
    if 'layers' in params:
        ds.setdefault('layers', []).extend(params.pop('layers').split(','))
    if 'styles' in params:
        ds.setdefault('styles', []).extend(params.pop('styles').split(','))

    params.pop('service', None)
    params.pop('request', None)

    ds['params'] = params

    # {x} placeholders shouldn't be encoded
    url = url.replace('%7B', '{')
    url = url.replace('%7D', '}')

    ds['url'] = url

    return ds


def _datasource_space_delimited(text):
    key_re = r'^\w+\s*=\s*'

    value_re = r'''(?x)
        " (?: \\. | [^"])* " |
        ' (?: \\. | [^'])* ' |
        \S+
    '''

    parens_re = r'\(.*?\)'

    def _cut(u, rx):
        m = re.match(rx, u)
        if not m:
            raise ValueError(f'datasource uri error, expected {rx!r}, found {u[:25]!r}')
        v = m.group(0)
        return v, u[len(v):].strip()

    def _unesc(s):
        return re.sub(r'\\(.)', '\1', s)

    def _mid(s):
        return s[1:-1].strip()

    def _value(v):
        if v.startswith(('\'', '\"')):
            return _unesc(_mid(v))
        return v

    ds = {}

    while text:
        # keyword=
        key, text = _cut(text, key_re)
        key = key.strip('= ')

        if key == 'sql':
            # 'sql=' is special and can contain whatever, it's always the last one
            ds[key] = text
            break

        elif key == 'table':
            # 'table=' is special, it can be `table="foo"` or `table="foo"."bar"` or table=`"foo"."bar" (geom)`
            v, text = _cut(text, value_re)
            ds['table'] = _value(v)

            if text.startswith('.'):
                v, text = _cut(text[1:], value_re)
                ds['table'] += '.' + _value(v)

            if text.startswith('('):
                v, text = _cut(text, parens_re)
                ds['geometryColumn'] = _mid(v)

        else:
            # just param=val
            v, text = _cut(text, value_re)
            ds[key] = _value(v)

    return ds


def _datasource_pipe_delimited(text):
    if '|' not in text:
        return {'path': text}

    path, rest = text.split('|', maxsplit=1)

    if '=' not in rest:
        return {'path': path, 'options': rest}

    ds = {'path': path}

    for p in rest.split('|'):
        k, v = p.split('=', maxsplit=1)
        ds[k] = v

    return ds


##


def parse_properties(el: gws.XmlElement):
    """Parse qgis property blocks.

    There are following forms:

    Scalar property::

        <WMSContactPhone type="QString">...

    Dict::

        <QFieldSync>
            <dirsToCopy type="QString">...
            <exportDirectoryProject type="QString">...
        </QFieldSync>

    Option map::

        <data-defined-properties>
            <Option type="Map">
                <Option type="QString" name="..." value="..."/>
                <Option name="properties"/>
          </Option>
        </data-defined-properties>


    """

    if not el:
        return {}

    _, val = _parse_property_tag(el)
    return val


def _parse_property_tag(el: gws.XmlElement):
    typ = el.get('type')
    name = el.tag
    is_opt = el.tag == 'Option'

    if is_opt and el.attr('name'):
        name = el.attr('name')

    if not typ or typ == 'Map':
        d = {}

        for c in el:
            k, v = _parse_property_tag(c)
            if k:
                d[k] = v

        if len(d) == 1 and 'Option' in d:
            return name, d['Option']

        return name, d

    if typ == 'List':
        ls = []
        for c in el:
            _, v = _parse_property_tag(c)
            ls.append(v)
        return name, ls

    if typ == 'QStringList':
        val = [c.text for c in el.findall('value')]
        return name, val

    if typ == 'QString':
        val = el.attr('value') if is_opt else el.text
        return name, val

    if typ == 'bool':
        val = el.attr('value') if is_opt else el.text
        return name, val.lower() == 'true'

    if typ == 'int':
        val = el.attr('value') if is_opt else el.text
        return name, _parse_int(val)

    if typ == 'double':
        val = el.attr('value') if is_opt else el.text
        return name, _parse_float(val)

    return '', None


##


def _extent_from_atts(el):
    # <spatial dimensions="2" miny="0" maxz="0" maxx="0" crs="EPSG:25832" minx="0" minz="0" maxy="0"/>

    return gws.gis.extent.from_list([
        el.get('minx'),
        el.get('miny'),
        el.get('maxx'),
        el.get('maxy'),
    ])


def _extent_from_tag(el):
    # <wgs84extent>
    #     <xmin>1</xmin>
    #     <ymin>2</ymin>
    #     <xmax>3</xmax>
    #     <ymax>4</ymax>
    # </wgs84extent>
    #
    # or
    #
    # <WMSExtent type="QStringList">
    #   <value>1</value>
    #   <value>2</value>
    #   <value>3</value>
    #   <value>4</value>
    # </WMSExtent>
    #

    if not el:
        return

    if el.attr('type') == 'QStringList':
        return gws.gis.extent.from_list([v.text for v in el.children()])

    if el.find('xmin'):
        return gws.gis.extent.from_list([
            el.textof('xmin'),
            el.textof('ymin'),
            el.textof('xmax'),
            el.textof('ymax'),
        ])


def _parse_msize(s):
    # e.g. 'position': '228.477,27.8455,mm'
    try:
        x, y, u = s.split(',')
        return float(x), float(y), u
    except Exception:
        return None


def _parse_int(s):
    try:
        return int(s)
    except Exception:
        return 0


def _parse_float(s):
    try:
        x = float(s)
    except Exception:
        return 0
    if math.isnan(x) or math.isinf(x):
        return 0
    return x
