"""QGIS Project layer, retains QGIS options and hierarchy."""

import PIL.Image
import io
import math

import gws
import gws.common.layer
import gws.common.layer.types
import gws.common.metadata
import gws.config.parser
import gws.tools.os2
import gws.gis.extent
import gws.gis.legend
import gws.gis.source
import gws.tools.net

import gws.types as t

from . import provider


class Config(gws.common.layer.ImageConfig):
    """QGIS Project layer"""

    directSearch: t.Optional[t.List[str]]  #: QGIS providers that should be searched directly
    path: t.FilePath  #: path to a qgs project file
    rootLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to use as roots
    excludeLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to exclude
    flattenLayers: t.Optional[gws.common.layer.types.FlattenConfig]  #: flatten the layer hierarchy
    layerConfig: t.Optional[t.List[gws.common.layer.CustomConfig]]  #: custom configurations for specific layers


class Object(gws.common.layer.Image):
    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.provider: provider.Object = provider.create_shared(self.root, self.config)
        self.own_crs = self.provider.supported_crs[0]

        self.meta = self.configure_metadata(self.provider.meta)
        self.title = self.meta.title

        self.direct_render = set(self.var('directRender', default=[]))
        self.direct_search = set(self.var('directSearch', default=[]))

        # by default, take the top-level layers as groups
        slf = self.var('rootLayers') or gws.gis.source.LayerFilter(level=1)
        self.root_layers = gws.gis.source.filter_layers(self.provider.source_layers, slf)
        self.exclude_layers = self.var('excludeLayers')
        self.flatten = self.var('flattenLayers')
        self.custom_layer_config = self.var('layerConfig', default=[])

        self.source_layers = []

        layer_cfgs = gws.compact(self._layer(sl, depth=1) for sl in self.root_layers)
        if gws.is_empty(layer_cfgs):
            raise gws.Error(f'no source layers in {self.uid!r}')

        top_group = {
            'type': 'group',
            'title': '',
            'layers': layer_cfgs
        }

        top_cfg = gws.config.parser.parse(top_group, 'gws.ext.layer.group.Config', strict=False, with_internal_objects=True)
        self.layers = gws.common.layer.add_layers_to_object(self, top_cfg.layers)

    @property
    def props(self):
        resolutions = set()
        for la in self.layers:
            resolutions.update(la.resolutions)

        return gws.merge(
            super().props,
            type='tree',
            url=gws.SERVER_ENDPOINT + '/cmd/mapHttpGetBox/layerUid/' + self.uid,
            layers=self.layers,
            resolutions=sorted(resolutions, reverse=True))

    def configure_legend(self):
        return super().configure_legend() or t.LayerLegend(enabled=True)

    def render_legend_image(self, context=None):
        paths = gws.compact(la.render_legend(context) for la in self.layers if la.has_legend)
        return gws.gis.legend.combine_legend_paths(paths)

    def render_box(self, rv, extra_params=None):
        return _render_box(self, rv, extra_params)

    def mapproxy_config(self, mc, options=None):
        # NB: qgis caps layers are always top-down

        source = mc.source({
            'type': 'wms',
            'supported_srs': self.provider.supported_crs,
            'forward_req_params': ['DPI__gws', 'LAYERS__gws'],
            'concurrent_requests': self.root.var('server.qgis.maxRequests', default=0),
            'req': {
                'url': self.provider.url,
                'map': self.provider.path,
                'transparent': True,
                'layers': '',
            },
            # add the file checksum to the config, so that the source and cache ids
            # in the mpx config are recalculated when the file changes
            '$hash': gws.tools.os2.file_checksum(self.provider.path)
        })

        self.mapproxy_layer_config(mc, source)

    def _layer(self, sl: t.SourceLayer, depth: int):
        if self.exclude_layers and gws.gis.source.layer_matches(sl, self.exclude_layers):
            return

        if sl.is_group:
            # NB use the absolute level to compute flatness, could also use relative (=depth)
            if self.flatten and sl.a_level >= self.flatten.level:
                la = self._flat_group_layer(sl)
            else:
                la = self._group_layer(sl, depth)
        else:
            la = self._image_layer(sl)

        if not la:
            return

        la = gws.merge(la, {
            'uid': gws.as_uid(sl.name),
            'title': sl.title,
            'clientOptions': {
                'visible': sl.is_visible,
                'expanded': sl.is_expanded,
            },
            'opacity': sl.opacity or 1,
        })

        if sl.scale_range:
            la['zoom'] = {
                'minScale': sl.scale_range[0],
                'maxScale': sl.scale_range[1],
            }

        p = self.var('templates')
        if p:
            la['templates'] = p

        custom = [gws.strip(c) for c in self.custom_layer_config if gws.gis.source.layer_matches(sl, c.applyTo)]
        if custom:
            la = gws.deep_merge(la, *custom)
            if la.applyTo:
                delattr(la, 'applyTo')

        return gws.compact(la)

    def _image_layer(self, sl: t.SourceLayer):
        prov = sl.data_source['provider']

        la = self._qgis_based_layer(sl)

        if not la:
            return

        if not sl.is_queryable:
            return la

        p = self._layer_search_provider(sl)
        if p:
            la['search'] = {
                'enabled': True,
                'providers': [p]
            }
        else:
            la['search'] = {
                'enabled': False
            }

        return la

    def _qgis_based_layer(self, sl: t.SourceLayer):
        self.source_layers.append(sl)
        return {
            'type': 'qgisleaf',
            '_layer': self,
            '_source_layers': [sl]
        }

    def _flat_group_layer(self, sl: t.SourceLayer):
        if self.flatten.useGroups:
            return {
                'type': 'qgisleaf',
                '_layer': self,
                '_source_layers': [sl]
            }

        ls = gws.gis.source.image_layers(sl)
        if ls:
            return {
                'type': 'qgisleaf',
                '_layer': self,
                '_source_layers': ls
            }

    def _layer_search_provider(self, sl: t.SourceLayer):
        p = self.var('search')
        if not p.enabled or p.providers:
            return

        ds = sl.data_source
        prov = ds['provider']

        if prov in self.direct_search:
            if prov == 'wms':
                return self._wms_search_provider(sl, ds)
            if prov == 'postgres':
                return self._postgres_search_provider(sl, ds)
            if prov == 'wfs':
                return self._wfs_search_provider(sl, ds)
            gws.log.warn(f'directSearch not supported for {prov!r}')

        # if no directSearch for this provider, then we make WMS request to the local QGIS

        return {
            'type': 'qgiswms',
            'uid': gws.as_uid(sl.name) + '_qgiswms_search',
            'path': self.path,
            'sourceLayers': {
                'names': [sl.name]
            }
        }

    def _wms_search_provider(self, sl, ds):
        return {
            'type': 'wms',
            'uid': gws.as_uid(sl.name) + '_wms_search',
            'url': self._make_wms_url(ds['url'], ds['params']),
            'sourceLayers': {
                'names': ds['layers'],
            }
        }

    def _postgres_search_provider(self, sl, ds):
        tab = sl.data_source['table']

        # 'table' can also be a select statement, in which case it might be enclosed in parens
        if tab.startswith('(') or tab.upper().startswith('SELECT '):
            return

        return {
            'type': 'qgispostgres',
            'uid': gws.as_uid(sl.name) + '_qgispostgres_search',
            'dataSource': ds
        }

    def _wfs_search_provider(self, sl, ds):
        cfg = {
            'type': 'wfs',
            'url': ds['url'],
            'uid': gws.as_uid(sl.name) + '_wfs_search',
        }
        if gws.get(ds, 'typeName'):
            cfg['sourceLayers'] = {
                'names': [ds['typeName']]
            }
        crs = gws.get(ds, 'params.srsname')
        inv = gws.get(ds, 'params.InvertAxisOrientation')
        if inv == '1' and crs:
            cfg['invertAxis'] = [crs]

        return cfg

    def _group_layer(self, sl: t.SourceLayer, depth: int):
        layers = gws.compact(self._layer(la, depth + 1) for la in sl.layers)
        if not layers:
            return
        return {
            'type': 'group',
            'title': sl.title,
            'uid': gws.as_uid(sl.name),
            'layers': layers
        }

    def _make_wms_url(self, url, params):
        # a wms url can be like "server?service=WMS....&bbox=.... &some-non-std-param=...
        # we need to keep non-std params for caps requests

        _std_params = {
            'service',
            'version',
            'request',
            'layers',
            'styles',
            'srs',
            'crs',
            'bbox',
            'width',
            'height',
            'format',
            'transparent',
            'bgcolor',
            'exceptions',
            'time',
            'sld',
            'sld_body',
        }
        p = {k: v for k, v in params.items() if k.lower() not in _std_params}
        return gws.tools.net.add_params(url, p)


# see also gws/common/layer/image.py
# need to copy this here to use qgis directly

def _render_box(layer: Object, rv: t.MapRenderView, extra_params: dict = None):
    extra_params = extra_params or {}

    if 'layers' in extra_params:
        source_names = []
        filters = []
        for uid in extra_params.pop('layers'):
            leaf = layer.root.find_by_uid(uid)
            for sl in leaf.source_layers:
                source_names.append(sl.name)
                if 'filter' in extra_params:
                    filters.append(sl.name + ': ' + extra_params['filter'])
                # if leaf.qgisfilter and sl.data_source and 'sql' in sl.data_source:
                #     filters.append(sl.name + ': ' + leaf.qgisfilter)

        extra_params['layers'] = source_names
        if filters:
            extra_params['filter'] = ';'.join(filters)


    # boxes larger than that will be tiled in _box_request
    size_threshold = 2500

    if not rv.rotation:
        return _box_request(layer, rv.bounds, rv.size_px[0], rv.size_px[1], extra_params, tile_size=size_threshold)

    # rotation: render a circumsquare around the wanted extent

    circ = gws.gis.extent.circumsquare(rv.bounds.extent)
    w, h = rv.size_px
    d = gws.gis.extent.diagonal((0, 0, w, h))

    r = _box_request(layer, t.Bounds(crs=rv.bounds.crs, extent=circ), d, d, extra_params, tile_size=size_threshold)
    if not r:
        return

    img: PIL.Image.Image = PIL.Image.open(io.BytesIO(r))

    # rotate the square (NB: PIL rotations are counter-clockwise)
    # and crop the square back to the wanted extent

    img = img.rotate(-rv.rotation, resample=PIL.Image.BICUBIC)
    img = img.crop((
        d / 2 - w / 2,
        d / 2 - h / 2,
        d / 2 + w / 2,
        d / 2 + h / 2,
    ))

    with io.BytesIO() as out:
        img.save(out, format='png')
        return out.getvalue()


def _box_request(layer: Object, bounds, width, height, params, tile_size):
    if width < tile_size and height < tile_size:
        return _get_map_request(layer, bounds, width, height, params)

    xcount = math.ceil(width / tile_size)
    ycount = math.ceil(height / tile_size)

    ext = bounds.extent

    bw = (ext[2] - ext[0]) * tile_size / width
    bh = (ext[3] - ext[1]) * tile_size / height

    grid = []

    for ny in range(ycount):
        for nx in range(xcount):
            e = [
                ext[0] + bw * nx,
                ext[3] - bh * (ny + 1),
                ext[0] + bw * (nx + 1),
                ext[3] - bh * ny,
            ]
            bounds = t.Bounds(crs=bounds.crs, extent=e)
            content = _get_map_request(layer, bounds, tile_size, tile_size, params)
            grid.append([nx, ny, content])

    out = PIL.Image.new('RGBA', (tile_size * xcount, tile_size * ycount), (0, 0, 0, 0))
    for nx, ny, content in grid:
        img = PIL.Image.open(io.BytesIO(content))
        out.paste(img, (nx * tile_size, ny * tile_size))

    out = out.crop((0, 0, width, height))

    buf = io.BytesIO()
    out.save(buf, 'PNG')
    return buf.getvalue()


def _get_map_request(layer: Object, bounds: t.Bounds, width, height, params: dict):
    ps = {
        'map': layer.provider.path,
        'bbox': bounds.extent,
        'width': width,
        'height': height,
        'crs': bounds.crs,
        'service': 'WMS',
        'request': 'GetMap',
        'version': '1.3.0',
        'format': 'image/png',
        'transparent': 'true',
        'styles': '',
    }
    ps.update(params)

    gws.log.debug(f'calling qgis: {ps!r}')

    resp = gws.tools.net.http_request(layer.provider.url, params=ps, timeout=0)
    if resp.content_type.startswith('image'):
        return resp.content
    raise ValueError(resp.text)
