import os
import re
import urllib.parse

from bs4 import BeautifulSoup
import lxml.etree

import gws
import gws.config
import gws.gis.source
import gws.types as t


def _get(node, path, default=None):
    for r in path.split('.'):
        if not node:
            break
        node = getattr(node, r)
    return node or default


def _parse_datasource_uri(uri):
    # see QGIS/src/core/qgsdatasourceuri.cpp

    rx = r'''(?x)
        (\w+ \s* = \s*)
        | " ( (?: \\. | [^"])* ) "
        | ' ( (?: \\. | [^'])* ) '
        | (\.)
        | \( (.+?) \)
        | (\S+)
    '''

    uri = uri.strip()
    ms = list(re.finditer(rx, uri))
    r = {}

    def _unesc(s):
        return re.sub(r'\\(.)', '\1', s)

    def _str(m):
        if m.group(6):
            return m.group(6)
        return _unesc(m.group(2) or m.group(3))

    while ms:
        # it must be a key
        key = ms.pop(0).group(1).strip('= ')

        if key == 'sql':
            # sql is the last
            if ms:
                r[key] = uri[ms[0].start():]
            break

        r[key] = _str(ms.pop(0))

        # table=xxx or table=xxx.yyy or table=xxx.yyy (geom)
        if key == 'table':
            if ms and ms[0].group(4):
                ms.pop(0)
                r[key] += '.' + _str(ms.pop(0))
            if ms and ms[0].group(5):
                r['geometryColumn'] = _unesc(ms.pop(0).group(5))

    return r


class SourceLayer(gws.gis.source.SourceLayer):
    pass


class ComposerItem:
    def __init__(self, tag, elem):
        self.tag = tag
        self.attrs = dict(elem.attrs)
        for e in elem:
            name = getattr(e, 'name', None)
            if name == 'ComposerItem':
                self.attrs.update(e.attrs)
            elif name:
                setattr(self, name, e.attrs)


_bigval = 1e10


def _fix_broken_xml(xml):
    # @TODO: there's a lot more to do here, e.g. isolate CDATA, fix other xml errors
    rx = r'&(?!amp;)'
    m = re.search(rx, xml)
    if not m:
        return xml
    return re.sub(rx, '&amp;', xml)


class Project:
    def __init__(self, path):
        self.path = path
        with open(self.path, 'rb') as fp:
            xml = fp.read().decode('utf8')

        # some QGIS versions have unescaped &'s in project files
        # BS passes recover=True down to lxml, so these &'s just get removed - not good!
        xml = _fix_broken_xml(xml)

        self.bs = BeautifulSoup(xml, 'lxml-xml')

        gws.log.info(f'loading "{path}"...')
        self._map_layers = self._parse_map_layers()
        self._all_layers = []
        self._root = self._parse_tree(self.bs.find('layer-tree-group'), '/')
        self._parse_group_values(self._root)

    @property
    def print_compositions(self):
        return [self._parse_print_composition(c) for c in self.bs.find_all('Composer')]

    @property
    def layers(self):
        return self._all_layers

    @property
    def top_layers(self):
        return self._root.layers

    def _parse_source(self, provider, source):
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
                p = urllib.parse.urlparse(url)
                if p.query:
                    params = {k.lower(): v for k, v in urllib.parse.parse_qsl(p.query)}
                    url = urllib.parse.urlunparse(p[:3] + ('', '', ''))

            return {'url': url, 'options': options, 'params': params, 'layers': layers}

        if provider in ('gdal', 'ogr'):
            return {'path': source}

        if provider == 'postgres':
            return gws.compact(_parse_datasource_uri(source))

        return {'source': source}

    def _parse_map_layers(self):
        ls = {}

        for p in self.bs.find_all('maplayer'):
            la = SourceLayer()
            la.uid = _get(p, 'id.text', 'layer_%d' % len(ls))
            la.title = _get(p, 'layername.text')

            s = _get(p, 'provider.text')
            if s:
                src = _get(p, 'datasource.text')
                la.data = {'provider': s, 'source': self._parse_source(s, src)}

            s = _get(p, 'srs.spatialrefsys.authid.text')
            if s:
                la.crs = s

            if _get(p, 'extent.xmax.text') is not None:
                la.extent = [
                    float(_get(p, 'extent.xmin.text', 0.0)),
                    float(_get(p, 'extent.ymin.text', 0.0)),
                    float(_get(p, 'extent.xmax.text', _bigval)),
                    float(_get(p, 'extent.ymax.text', _bigval)),
                ]

            if p.get('minimumScale') is not None and p.get('hasScaleBasedVisibilityFlag') == '1':
                m = p['minimumScale']
                if m == 'inf':
                    m = 0.0
                la.min_scale = max(0.0, float(m))
                la.max_scale = min(_bigval, float(p['maximumScale']))

            s = _get(p, 'layerTransparency.text')
            if s is not None:
                la.opacity = 1 if s == '0' else (100 - int(s)) / 100

            ls[la.uid] = la

        return ls

    def _parse_tree(self, node, path):
        tag = node.name
        if tag not in ('layer-tree-group', 'layer-tree-layer'):
            return

        visible = node.get('checked') != 'Qt::Unchecked'

        if tag == 'layer-tree-group':
            name = node.get('name') or ''
            uid = gws.as_uid(name)
            p = os.path.join(path, uid)

            la = SourceLayer()
            la.path = p
            la.type = 'group'
            la.title = name
            la.uid = uid
            la.visible = visible

            self._all_layers.append(la)

            la.layers = gws.compact(self._parse_tree(s, p) for s in node)
            return la

        if tag == 'layer-tree-layer':
            uid = node['id']
            la = self._map_layers[uid]
            la.path = os.path.join(path, gws.as_uid(uid))
            la.type = 'layer'
            la.visible = visible
            self._all_layers.append(la)
            return la

    def _parse_group_values(self, la):
        if la.type != 'group':
            return

        la.crs = None
        la.extent = [_bigval, _bigval, 0.0, 0.0]
        la.min_scale = _bigval
        la.max_scale = 0.0

        for sub in la.layers:
            self._parse_group_values(sub)

            if sub.crs:
                la.crs = sub.crs
            if sub.extent:
                la.extent = [
                    min(la.extent[0], sub.extent[0]),
                    min(la.extent[1], sub.extent[1]),
                    max(la.extent[2], sub.extent[2]),
                    max(la.extent[3], sub.extent[3]),
                ]
            if sub.max_scale:
                la.min_scale = min(la.min_scale, sub.min_scale)
                la.max_scale = max(la.max_scale, sub.max_scale)

    def _parse_print_composition(self, composer):
        composition = composer.find('Composition')

        classes = {name: cls for name, cls in globals().items() if name.startswith('Composer')}
        items = gws.compact(self._parse_composer_item(e, classes) for e in composition)

        return t.Data({
            'attrs': dict(composition.attrs),
            'items': items,
            'title': composer['title'],
        })

    def _parse_composer_item(self, elem, classes):
        tag = getattr(elem, 'name', None)
        if not tag or not tag.startswith('Composer'):
            return
        return ComposerItem(tag, elem)


def parse_project(path, cache=False):
    if not cache:
        return Project(path)
    return gws.get_global('qgis.' + path, lambda: Project(path))
