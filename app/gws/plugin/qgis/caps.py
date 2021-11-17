import math
import re
import urllib.parse

import gws
import gws.gis.crs
import gws.lib.metadata
import gws.lib.net
import gws.gis.ows.parseutil as u
import gws.lib.xml3 as xml3
import gws.types as t

_bigval = 1e10


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
    metadata: gws.IMetadata
    print_templates: t.List[PrintTemplate]
    properties: dict
    source_layers: t.List[gws.SourceLayer]
    project_crs: t.Optional[gws.ICrs]
    version: str


def parse(xml: str) -> Caps:
    root_el = xml3.from_string(xml, sort_atts=True, strip_ns=True, to_lower=True)

    ver = root_el.attributes.get('version', '').split('-')[0]
    if not ver.startswith('3'):
        raise gws.Error(f'unsupported QGIS version {ver!r}')

    caps = Caps(version=ver)

    caps.properties = _properties(xml3.first(root_el, 'properties'))
    caps.metadata = _project_meta_from_props(caps.properties)
    caps.project_crs = gws.gis.crs.get(xml3.text(root_el, 'projectCrs.spatialrefsys.authid'))
    caps.print_templates = _layouts(root_el)

    map_layers = _map_layers(root_el, caps.properties)
    root_group = _map_layer_tree(xml3.first(root_el, 'layer-tree-group'), map_layers)
    caps.source_layers = u.enum_source_layers(root_group.layers)

    return caps


##

def _properties(el: gws.XmlElement):
    # parse the nested `properties` structure into a flat dict
    # the struct is like this:
    #
    # <properties>
    #       ...
    #       <PositionPrecision>
    #            <Automatic type="bool">true</Automatic>
    #            <DecimalPlaces type="int">2</DecimalPlaces>
    #            ...
    #

    # <WMSFees type="QString">conditions unknown</WMSFees>
    # <WMSImageQuality type="int">90</WMSImageQuality>
    # <WMSKeywordList type="QStringList">
    #   <value>one</value>
    #   <value>two</value>
    #   <value>three</value>
    # </WMSKeywordList>

    ty = xml3.attr(el, 'type')

    if not ty:
        return gws.strip({c.name.lower(): _properties(c) for c in el.children})

    # e.g.
    # <WMSKeywordList type="QStringList">
    #   <value>A</value>
    #   <value>B</value>
    #   <value>C</value>

    if ty == 'QStringList':
        return xml3.text_list(el)
    if ty == 'QString':
        return el.text
    if ty == 'bool':
        return el.text.lower() == 'true'
    if ty == 'int':
        return _parse_int(el.text)
    if ty == 'double':
        return _parse_float(el.text)


def _project_meta_from_props(props):
    # @TODO should also read `properties.projectMetadata`

    d = gws.strip({
        'abstract': _pval(props, 'WMSServiceAbstract'),
        'attribution': _pval(props, 'CopyrightLabel.Label'),
        'keywords': _pval(props, 'WMSKeywordList'),
        'title': _pval(props, 'WMSServiceTitle'),
        'contactEmail': _pval(props, 'WMSContactMail'),
        'contactOrganization': _pval(props, 'WMSContactOrganization'),
        'contactPerson': _pval(props, 'WMSContactPerson'),
        'contactPhone': _pval(props, 'WMSContactPhone'),
        'contactPosition': _pval(props, 'WMSContactPosition'),
    })
    return gws.lib.metadata.from_dict(d)


##


def _map_layers(root_el: gws.XmlElement, props) -> t.Dict[str, gws.SourceLayer]:
    no_wms_layers = set(_pval(props, 'WMSRestrictedLayers') or [])
    use_layer_ids = _pval(props, 'WMSUseLayerIDs')

    map_layers = {}

    for el in xml3.all(root_el, 'projectlayers.maplayer'):
        sl = _map_layer(el)

        if not sl:
            continue

        title = sl.metadata.get('title')

        # no_wms_layers always contains titles, not ids (=names)
        if title in no_wms_layers:
            continue

        uid = xml3.text(el, 'id')
        if use_layer_ids:
            name = uid
        else:
            name = xml3.text(el, 'shortname') or xml3.text(el, 'layername')

        sl.title = title
        sl.name = name
        sl.metadata.set('name', name)

        map_layers[uid] = sl

    return map_layers


def _map_layer(layer_el: gws.XmlElement):
    sl = gws.SourceLayer()

    sl.metadata = _map_layer_metadata(layer_el)

    sl.supported_bounds = []

    crs = gws.gis.crs.get(xml3.text(layer_el, 'srs.spatialrefsys.authid'))
    ext = xml3.first(layer_el, 'extent')

    if crs and ext:
        sl.supported_bounds.append(gws.Bounds(
            crs=crs,
            extent=(
                _parse_float(xml3.text(ext, 'xmin')),
                _parse_float(xml3.text(ext, 'ymin')),
                _parse_float(xml3.text(ext, 'xmax')),
                _parse_float(xml3.text(ext, 'ymax')),
            )
        ))

    if layer_el.attributes.get('hasScaleBasedVisibilityFlag') == '1':
        # in qgis, maxScale < minScale
        a = _parse_float(layer_el.attributes.get('maxScale'))
        z = _parse_float(layer_el.attributes.get('minScale'))
        if z > a:
            sl.scale_range = [a, z]

    prov = xml3.text(layer_el, 'provider').lower()
    ds = _parse_datasource(prov, xml3.text(layer_el, 'datasource'))
    if ds and 'provider' not in ds:
        ds['provider'] = prov
    sl.data_source = ds

    s = xml3.text(layer_el, 'layerOpacity')
    if s:
        sl.opacity = _parse_float(s)

    s = xml3.text(layer_el, 'flags.Identifiable')
    sl.is_queryable = s == '1'

    return sl


def _map_layer_metadata(layer_el: gws.XmlElement):
    def tx(s):
        return xml3.text(layer_el, s)

    d = gws.strip({
        'abstract': tx('resourceMetadata.abstract'),
        'contactEmail': tx('email'),
        'contactFax': tx('fax'),
        'contactOrganization': tx('organization'),
        'contactPerson': tx('name'),
        'contactPhone': tx('voice'),
        'contactPosition': tx('position'),
        'contactRole': tx('role'),
        'keywords': xml3.text_list(layer_el, 'keywordList.value'),
        'name': tx('id'),
        'title': tx('layername'),
        'url': tx('metadataUrl'),
    })

    return gws.lib.metadata.from_dict(d) if d else None


# layer trees:

# <layer-tree-group>
#     <layer-tree-group checked="Qt::Checked" expanded="1" name="...">
#         <layer-tree-layer ... checked="Qt::Checked" expanded="1" id="...">
#         ...


def _map_layer_tree(el: gws.XmlElement, map_layers):
    visible = el.attributes.get('checked') != 'Qt::Unchecked'
    expanded = el.attributes.get('expanded') == '1'

    if el.name == 'layer-tree-group':
        title = el.attributes.get('name')
        # qgis doesn't write 'id' for groups but our generators might
        name = el.attributes.get('id') or title

        sl = gws.SourceLayer(title=title, name=name)
        sl.metadata = gws.lib.metadata.from_args(title=title, name=name)

        sl.is_visible = visible
        sl.is_expanded = expanded
        sl.is_group = True
        sl.is_queryable = False
        sl.is_image = False

        sl.layers = gws.compact(_map_layer_tree(c, map_layers) for c in el.children)
        return sl

    if el.name == 'layer-tree-layer':
        sl = map_layers.get(el.attributes.get('id'))
        if sl:
            sl.is_visible = visible
            sl.is_expanded = expanded
            sl.is_group = False
            sl.is_image = True
            return sl


##

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
}


def _layouts(root_el: gws.XmlElement):
    tpls = []

    for layout_el in xml3.all(root_el, 'Layouts.Layout'):
        tpl = PrintTemplate(
            title=layout_el.attributes.get('name', ''),
            attributes=layout_el.attributes,
            index=len(tpls),
            elements=[],
        )
        pc_el = xml3.first(layout_el, 'PageCollection')
        if pc_el:
            tpl.elements.extend(gws.compact(_layout_element(c) for c in pc_el.children))
        tpl.elements.extend(gws.compact(_layout_element(c) for c in layout_el.children))

        tpls.append(tpl)

    return tpls


def _layout_element(item_el):
    type = _LAYOUT_TYPES.get(_parse_int(item_el.attributes.get('type')))
    uuid = item_el.attributes.get('uuid')
    if type and uuid:
        return PrintTemplateElement(
            type=type,
            uuid=uuid,
            attributes=item_el.attributes,
            position=_parse_msize(item_el.attributes.get('position')),
            size=_parse_msize(item_el.attributes.get('size')),
        )


##


def _parse_datasource(provider, source):
    if provider == 'wfs':
        params = _parse_datasource_uri(source)
        url = params.pop('url', '')
        if not url:
            return {}
        p = gws.lib.net.parse_url(url)
        typename = params.pop('typename', '') or p.params.get('typename')
        return {
            'url': url,
            'typeName': typename,
            'params': params
        }

    if provider == 'wms':
        options = {}
        for k, v in urllib.parse.parse_qs(source).items():
            options[k] = v[0] if len(v) < 2 else v

        layers = []
        if 'layers' in options:
            layers = options.pop('layers')
            # 'layers' must be a list
            if isinstance(layers, str):
                layers = [layers]

        url = options.pop('url', '')
        params = {}
        if url:
            url, params = _parse_url_with_qs(url)

        d = {'url': url, 'options': options, 'params': params, 'layers': layers}
        if 'tileMatrixSet' in options:
            d['provider'] = 'wmts'
        return d

    if provider in ('gdal', 'ogr'):
        return {'path': source}

    if provider == 'postgres':
        return gws.compact(_parse_datasource_uri(source))

    return {'source': source}


def _parse_url_with_qs(url):
    p = urllib.parse.urlparse(url)
    params = {}
    if p.query:
        params = {k: v for k, v in urllib.parse.parse_qsl(p.query)}
    url = urllib.parse.urlunparse(p[:3] + ('', '', ''))
    return url, params


def _parse_datasource_uri(uri):
    # see QGIS/src/core/qgsdatasourceuri.cpp... ;(
    #
    # the format appears to be key = value pairs, where value can be quoted and c-escaped
    # 'table=' is special, is can be table="foo" or table="foo"."bar" or table="foo"."bar" (geom)
    # 'sql=' is special too and can contain whatever, it's always the last one
    #
    # alternatively, a datasource can be an url

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

    def _parse(u, r):
        while u:
            # keyword=
            key, u = _cut(u, r'^\w+\s*=\s*')
            key = key.strip('= ')

            # sql=rest...
            if key == 'sql':
                r[key] = u
                break

            elif key == 'table':
                # table=schema.tab(geom)
                v1, u = _cut(u, value_re)
                v1 = _value(v1)

                v2 = v3 = ''

                if u.startswith('.'):
                    v2, u = _cut(u[1:], value_re)
                    v2 = _value(v2)

                if u.startswith('('):
                    v3, u = _cut(u, parens_re)
                    v3 = _mid(v3)

                r['table'] = (v1 + '.' + v2) if v2 else v1
                if v3:
                    r['geometryColumn'] = v3

            else:
                # just param=val
                v, u = _cut(u, value_re)
                r[key] = _value(v)

    if uri.startswith(('http://', 'https://')):
        return {'url': uri}

    rec = {}
    _parse(uri, rec)
    return rec


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
