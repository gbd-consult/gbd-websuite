"""QGIS project xml parser."""

import math
import re

import gws
import gws.gis.crs
import gws.lib.metadata
import gws.lib.net
import gws.lib.xmlx as xmlx
import gws.types as t


class PrintTemplateElement(gws.Data):
    type: str
    uuid: str
    attributes: dict
    position: gws.MPoint
    size: gws.MSize


class PrintTemplate(gws.Data):
    title: str
    index: int
    attributes: dict
    elements: t.List[PrintTemplateElement]


class Caps(gws.Data):
    metadata: gws.Metadata
    printTemplates: t.List[PrintTemplate]
    projectCrs: gws.ICrs
    properties: dict
    sourceLayers: t.List[gws.SourceLayer]
    version: str


def parse(xml: str) -> Caps:
    el = gws.lib.xmlx.from_string(xml)
    return parse_element(el)


def parse_element(root_el: gws.IXmlElement) -> Caps:
    caps = Caps()

    caps.version = root_el.get('version')
    caps.properties = _properties(root_el)
    caps.metadata = _project_metadata(root_el)
    caps.print_templates = _print_templates(root_el)

    caps.projectCrs = gws.gis.crs.get(
        root_el.text_of('projectCrs/spatialrefsys/authid') or '4326')

    layers_dct = _map_layers(root_el, caps.properties)
    root_group = _layer_tree(root_el.find('layer-tree-group'), layers_dct)
    caps.sourceLayers = gws.gis.source.check_layers(root_group.layers)

    return caps


##

def _properties(root_el: gws.IXmlElement):
    el = root_el.find('properties')
    if not el:
        return {}
    return _props(el)


def _props(el):
    # parse the nested `properties` structure into a flat dict
    # each child of 'properties' is either a structure (without type=)
    # or a value (with type=)

    typ = el.get('type')
    if not typ:
        return gws.strip({c.tag: _props(c) for c in el})

    if typ == 'QStringList':
        return el.text_list('value')
    if typ == 'QString':
        return el.text
    if typ == 'bool':
        return el.text.lower() == 'true'
    if typ == 'int':
        return _parse_int(el.text)
    if typ == 'double':
        return _parse_float(el.text)


##


def _project_metadata(root_el) -> gws.Metadata:
    md = gws.Metadata()

    el = root_el.find('projectMetadata')
    if el:
        _metadata(el, md)

    # @TODO supplementary metadata
    return md


def _layer_metadata(layer_el) -> gws.Metadata:
    # Layer metadata is either Layer->Properties->Metadata (stored in maplayer/resourceMetadata),
    # or Layer->Properties->QGIS Server (stored directly under maplayer/abstract, maplayer/keywordList and so on.

    md = gws.Metadata()

    el = layer_el.find('resourceMetadata')
    if el:
        _metadata(el, md)

    # @TODO supplementary metadata
    return md


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


def _metadata(el: gws.IXmlElement, md: gws.Metadata):
    # extract metadata from projectMetadata/resourceMetadata
    _add_dict(md, el.text_dict(), _meta_mapping)

    md.keywords = []
    for kw in el.findall('keywords'):
        keywords = kw.text_list('keyword')
        if kw.get('vocabulary') == 'gmd:topicCategory':
            md.isoTopicCategories = keywords
        else:
            md.keywords.extend(keywords)

    contact_el = el.find('contact')
    if contact_el:
        _add_dict(md, contact_el.text_dict(), _contact_mapping)
        addr_el = contact_el.find('contactAddress')
        if addr_el:
            # NB we only support one contact address
            _add_dict(md, addr_el.text_dict(), _contact_mapping)

    md.metaLinks = []
    for e in el.findall('links/link'):
        # @TODO clarify
        md.metaLinks.append(gws.MetadataLink(
            url=e.get('url'),
            description=e.get('description'),
            mimeType=e.get('mimeType'),
            format=e.get('format'),
            name=e.get('name'),
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

    e = el.find('extent/spatial')
    if e:
        md.bounds = gws.Bounds(
            crs=gws.gis.crs.get(e.get('crs')),
            extent=(
                _parse_float(e.get('minx')),
                _parse_float(e.get('miny')),
                _parse_float(e.get('maxx')),
                _parse_float(e.get('maxy')),
            )
        )

    e = el.find('extent/temporal')
    if e:
        md.dateBegin = e.text_of('period/start')
        md.dateEnd = e.text_of('period/end')


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


def _print_templates(root_el: gws.IXmlElement):
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
            tpl.elements.extend(gws.compact(_layout_element(c) for c in pc_el))

        tpl.elements.extend(gws.compact(_layout_element(c) for c in layout_el))

        templates.append(tpl)

    return templates


def _layout_element(item_el: gws.IXmlElement):
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


def _map_layers(root_el: gws.IXmlElement, properties) -> t.Dict[str, gws.SourceLayer]:
    no_wms_layers = set(properties.get('WMSRestrictedLayers', []))
    use_layer_ids = properties.get('WMSUseLayerIDs', False)

    map_layers = {}

    for el in root_el.findall('projectlayers/maplayer'):
        sl = _map_layer(el)
        if not sl:
            continue

        title = sl.metadata.get('title')

        # no_wms_layers always contains titles, not ids (=names)
        if title in no_wms_layers:
            continue

        uid = el.text_of('id')
        if use_layer_ids:
            name = uid
        else:
            name = el.text_of('shortname') or el.text_of('layername')

        sl.title = title
        sl.name = name
        # sl.metadata.set('name', name)

        map_layers[uid] = sl

    return map_layers


def _map_layer(layer_el: gws.IXmlElement):
    sl = gws.SourceLayer(
        supportedBounds=[],
        supportedCrs=[],
    )

    sl.metadata = _layer_metadata(layer_el)

    crs = gws.gis.crs.get(layer_el.text_of('srs/spatialrefsys/authid'))
    ext = layer_el.find('extent')

    if crs and ext:
        sl.supportedBounds.append(gws.SourceBounds(crs=crs, extent=_parse_extent(ext)))

    ext = layer_el.find('wgs84extent')
    if ext:
        sl.wgsExtent = _parse_extent(ext)

    if layer_el.get('hasScaleBasedVisibilityFlag') == '1':
        # in qgis, maxScale < minScale
        a = _parse_float(layer_el.get('maxScale'))
        z = _parse_float(layer_el.get('minScale'))
        if z > a:
            sl.scaleRange = [a, z]

    prov = layer_el.text_of('provider').lower()
    ds = _parse_datasource(prov, layer_el.text_of('datasource'))

    gws.p(layer_el.text_of('datasource'), ds)

    if ds and 'provider' not in ds:
        # wmts and xyz are both 'wms' in qgis
        if prov == 'wms' and 'tileMatrixSet' in ds:
            prov = 'wmts'
        if prov == 'wms' and ds.get('type') == 'xyz':
            prov = 'xyz'
        ds['provider'] = prov

    sl.dataSource = ds

    s = layer_el.text_of('layerOpacity')
    if s:
        sl.opacity = _parse_float(s)

    s = layer_el.text_of('flags/Identifiable')
    sl.isQueryable = s == '1'

    return sl


# layer trees:

# <layer-tree-group>
#     <layer-tree-group checked="Qt::Checked" expanded="1" name="...">
#         <layer-tree-layer ... checked="Qt::Checked" expanded="1" id="...">
#         ...


def _layer_tree(el: gws.IXmlElement, layers_dct):
    visible = el.get('checked') != 'Qt::Unchecked'
    expanded = el.get('expanded') == '1'

    if el.tag == 'layer-tree-group':
        title = el.get('name')
        # qgis doesn't write 'id' for groups but our generators might
        name = el.get('id') or title

        sl = gws.SourceLayer(title=title, name=name)
        sl.metadata = {}  # gws.lib.metadata.from_args(title=title, name=name)

        sl.isVisible = visible
        sl.isExpanded = expanded
        sl.isGroup = True
        sl.isQueryable = False
        sl.isImage = False

        sl.layers = gws.compact(_layer_tree(c, layers_dct) for c in el)
        return sl

    if el.tag == 'layer-tree-layer':
        sl = layers_dct.get(el.get('id'))
        if sl:
            sl.isVisible = visible
            sl.isExpanded = expanded
            sl.isGroup = False
            sl.isImage = True
            return sl


##


##


def _parse_datasource(provider, text):
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
        # ../rel/path/test.gpkg|layername=name
        if '|' not in text:
            return {'path': text}
        path, opt = text.split('|', maxsplit=1)
        if '=' not in opt:
            return {'path': path, 'options': opt}
        k, v = opt.split('=', maxsplit=1)
        return {'path': path, k: v}

    return {'text': text}


_array_opts = {'layers', 'styles'}


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

    u = gws.lib.net.parse_url(ds['url'])
    params = u.params

    if 'typename' in params:
        ds['typename'] = params.pop('typename')
    if 'layers' in params:
        ds.setdefault('layers', []).extend = params.pop('layers').split(',')
    if 'styles' in params:
        ds.setdefault('styles', []).extend = params.pop('styles').split(',')

    params.pop('service', None)
    params.pop('request', None)

    ds['params'] = params

    u.params = {}
    url = gws.lib.net.make_url(u)

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
            # 'table=' is special, is can be table="foo" or table="foo"."bar" or table="foo"."bar" (geom)
            v1, text = _cut(text, value_re)
            v1 = _value(v1)

            v2 = v3 = ''

            if text.startswith('.'):
                v2, text = _cut(text[1:], value_re)
                v2 = _value(v2)

            if text.startswith('('):
                v3, text = _cut(text, parens_re)
                v3 = _mid(v3)

            ds['table'] = (v1 + '.' + v2) if v2 else v1
            if v3:
                ds['geometryColumn'] = v3

        else:
            # just param=val
            v, text = _cut(text, value_re)
            ds[key] = _value(v)

    return ds


def _parse_extent(extent_el):
    return (
        _parse_float(extent_el.text_of('xmin')),
        _parse_float(extent_el.text_of('ymin')),
        _parse_float(extent_el.text_of('xmax')),
        _parse_float(extent_el.text_of('ymax')),
    )


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


def _pval(props, key):
    return gws.get(props, key.lower())
