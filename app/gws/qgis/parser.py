import re
import math
import urllib.parse

import gws
import gws.types as t
import gws.tools.xml3
import gws.tools.net
import gws.ows.parseutil as u

from . import types

_bigval = 1e10


def parse(srv: t.ServiceInterface, xml):
    root = gws.tools.xml3.from_string(xml)

    srv.properties = _properties(root.first('properties'))
    srv.meta = _project_meta_from_props(srv.properties)
    srv.version = root.attr('version', '').split('-')[0]

    if srv.version.startswith('2'):
        srv.print_templates = _print_v2(root)
    if srv.version.startswith('3'):
        srv.print_templates = _print_v3(root)

    for n, cc in enumerate(srv.print_templates):
        cc.index = n

    map_layers = _map_layers(root, srv.properties)

    root_group = _tree(root.first('layer-tree-group'), map_layers)
    srv.layers = u.flatten_source_layers(root_group.layers)

    crs = None

    if srv.version.startswith('2'):
        crs = _pval(srv.properties, 'SpatialRefSys.ProjectCrs')
    if srv.version.startswith('3'):
        crs = root.get_text('projectCrs.spatialrefsys.authid')

    if crs:
        srv.supported_crs = [crs]


def _project_meta_from_props(props):
    p = gws.strip({
        'abstract': _pval(props, 'WMSServiceAbstract'),
        'attribution': _pval(props, 'CopyrightLabel.Label'),
        'keywords': _pval(props, 'WMSKeywordList'),
        'title': _pval(props, 'WMSServiceTitle'),
    })

    if not p:
        return

    meta = t.MetaData(p)

    p = gws.strip({
        'email': _pval(props, 'WMSContactMail'),
        'organization': _pval(props, 'WMSContactOrganization'),
        'person': _pval(props, 'WMSContactPerson'),
        'phone': _pval(props, 'WMSContactPhone'),
        'position': _pval(props, 'WMSContactPosition'),

    })
    if p:
        meta.contact = t.MetaContact(p)

    return meta


def _pval(props, key):
    return gws.get(props, key.lower())


def _properties(el):
    # NB: property keys converted to lowercase, cased args to _pval are just for readability

    et = el.attr('type')

    if not et:
        return gws.strip({e.name.lower(): _properties(e) for e in el.all()})

    if et == 'int':
        return int(el.text or '0')

    if et == 'QStringList':
        return gws.strip([e.text for e in el.all()])

    if et == 'QString':
        return el.text

    if et == 'bool':
        return el.text.lower() == 'true'

    if et == 'double':
        return _float(el.text)


def _tree(el, map_layers):
    visible = el.attr('checked') != 'Qt::Unchecked'
    expanded = el.attr('expanded') == '1'

    if el.name == 'layer-tree-group':
        sl = types.SourceLayer()
        n = el.attr('name')
        sl.meta.title = sl.meta.name = sl.title = sl.name = n

        sl.is_visible = visible
        sl.is_expanded = expanded
        sl.is_group = True
        sl.is_queryable = False
        sl.is_image = False

        sl.layers = gws.compact(_tree(e, map_layers) for e in el.all())
        return sl

    if el.name == 'layer-tree-layer':
        sl = map_layers.get(el.attr('id'))
        if sl:
            sl.is_visible = visible
            sl.is_expanded = expanded
            sl.is_group = False
            sl.is_image = True
            return sl


def _map_layers(root, props):
    disabled_layers = set(_pval(props, 'Identify.disabledLayers') or [])
    no_wms_layers = set(_pval(props, 'WMSRestrictedLayers') or [])
    use_layer_ids = _pval(props, 'WMSUseLayerIDs')

    map_layers = {}

    for el in root.all('projectlayers.maplayer'):
        sl = _map_layer(el)

        if not sl:
            continue

        # no_wms_layers always contains titles, not ids (=names)

        if sl.meta.title in no_wms_layers:
            continue

        # ggis2: non-queryable layers are on the identify.disabledlayers list
        # ggis3: non-queryable layers have <flags><Identifiable>0

        s = el.get_text('flags.Identifiable')
        if s == '1':
            sl.is_queryable = True
        elif s == '0':
            sl.is_queryable = False
        else:
            sl.is_queryable = sl.meta.name not in disabled_layers

        sl.title = sl.meta.title
        sl.name = sl.meta.name if use_layer_ids else sl.title

        map_layers[sl.meta.name] = sl

    return map_layers


def _layer_meta(el):
    p = gws.strip({
        'abstract': el.get_text('resourceMetadata.abstract'),
        'keywords': gws.compact(e.text for e in el.all('keywordList.value')),
        'title': el.get_text('layername'),
        'name': el.get_text('id'),
        'url': el.get_text('metadataUrl'),
    })
    if not p:
        return

    meta = t.MetaData(p)

    p = gws.strip({
        k: el.get_text('resourceMetadata.contact.' + k)
        for k in ('name', 'organization', 'position', 'voice', 'fax', 'email', 'role')
    })

    if p:
        meta.contact = t.MetaContact(p)

    return meta


def _map_layer(el):
    sl = types.SourceLayer()

    sl.meta = _layer_meta(el)

    crs = el.get_text('srs.spatialrefsys.authid')

    sl.supported_crs = [crs]
    sl.extents = {}

    e = el.first('extent')
    if e:
        sl.extents[crs] = [
            _float(e.get_text('xmin')),
            _float(e.get_text('ymin')),
            _float(e.get_text('xmax')),
            _float(e.get_text('ymax')),
        ]

    if el.attr('hasScaleBasedVisibilityFlag') == '1':
        # sic! these are called min-max in qgis2 and max-min in qgis3
        a = _float(el.attr('minimumScale') or el.attr('maxScale'))
        z = _float(el.attr('maximumScale') or el.attr('minScale'))
        if z > a:
            sl.scale_range = [a, z]

    prov = el.get_text('provider').lower()
    ds = _data_source(prov, el.get_text('datasource'))
    if 'provider' not in ds:
        ds['provider'] = prov
    sl.data_source = ds

    s = el.get_text('layerTransparency')
    if s:
        sl.opacity = 1 if s == '0' else (100 - int(s)) / 100

    s = el.get_text('layerOpacity')
    if s:
        sl.opacity = _float(s)

    return sl


"""
print templates in qgis-2:

<Composer title="..."
   <Composition ...
      <ComposerPicture elementAttrs...
          <ComposerItem itemAttrs...
          other tags
      <ComposerArrow elementAttrs...
          <ComposerItem itemAttrs...
          other tag

<Composer title="..."
      etc

we merge elementAttrs+itemAttrs and ignore other tags within elements

"""


def _print_v2(root):
    return [_print_v2_composer(el) for el in root.all('Composer')]


def _print_v2_composer(composer):
    oo = types.PrintTemplate()

    oo.title = composer.attr('title', '')

    composition = composer.first('Composition')
    oo.attrs = _lower_attrs(composition)

    oo.elements = [
        _print_v2_element(el)
        for el in composition.all()
        if el.name.startswith('Composer')
    ]

    return oo


def _print_v2_element(el):
    oo = types.PrintTemplateElement()
    oo.type = el.name[len('Composer'):].lower()
    oo.attrs = _lower_attrs(el)

    for item in el.all():
        if item.name == 'ComposerItem':
            oo.attrs.update(_lower_attrs(item))

    return oo


"""
print templates in qgis-3:

<Layouts>
   <Layout name="..." ...
      <PageCollection...
          <LayoutItem ....
      <LayoutItem type="<int, see below>" ...
          other tags
      <LayoutItem type="<int>" ...
          other tags

   <Layout name="..." ...
     etc

"""

# see QGIS3/QGIS/src/core/layout/qgslayoutitemregistry.h

_QGraphicsItem_UserType = 65536  # https://doc.qt.io/qtforpython/PySide2/QtWidgets/QGraphicsItem.html

_COMP3_LAYOUT_TYPE_FIRST = _QGraphicsItem_UserType + 100

_COMP3_LAYOUT_TYPES = {
    _COMP3_LAYOUT_TYPE_FIRST + n: s for n, s in enumerate(
    [
        'LayoutItem',
        'LayoutGroup',
        'LayoutPage',
        'LayoutMap',
        'LayoutPicture',
        'LayoutLabel',
        'LayoutLegend',
        'LayoutShape',
        'LayoutPolygon',
        'LayoutPolyline',
        'LayoutScaleBar',
        'LayoutFrame',
        'LayoutHtml',
        'LayoutAttributeTable',
        'LayoutTextTable',
    ])
}


def _print_v3(root):
    return [_print_v3_layout(el) for el in root.all('Layouts.Layout')]


def _print_v3_layout(layout):
    oo = types.PrintTemplate()
    oo.title = layout.attr('name', '')
    oo.attrs = _lower_attrs(layout)

    oo.elements = [_print_v3_item(item) for item in layout.all('PageCollection.LayoutItem')]
    oo.elements.extend(_print_v3_item(item) for item in layout.all('LayoutItem'))

    return oo


def _print_v3_item(el):
    oo = types.PrintTemplateElement()
    oo.type = _COMP3_LAYOUT_TYPES[int(el.attr('type'))][len('Layout'):].lower()
    oo.attrs = _lower_attrs(el)
    return oo


def _data_source(provider, source):
    if provider == 'wfs':
        params = _parse_datasource_uri(source)
        url = params.pop('url', '')
        if not url:
            return {}
        p = gws.tools.net.parse_url(url)
        typename = params.pop('typename', '') or p['params'].get('typename')
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


def _lower_attrs(el):
    return {k.lower(): v for k, v in el.attr_dict.items()}


def _float(s):
    try:
        x = float(s)
    except:
        return 0
    if math.isnan(x) or math.isinf(x):
        return 0
    return x
