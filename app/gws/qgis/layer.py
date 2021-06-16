"""QGIS Project layer, retains QGIS options and hierarchy."""

import gws
import gws.base.layer
import gws.base.layer.types
import gws.base.metadata
import gws.config.parser
import gws.gis.extent
import gws.gis.legend
import gws.gis.source
import gws.lib.net

import gws.types as t

from . import provider


class Config(gws.base.layer.ImageConfig):
    """QGIS Project layer"""

    directRender: t.Optional[t.List[str]]  #: QGIS providers that should be rendered directly
    directSearch: t.Optional[t.List[str]]  #: QGIS providers that should be searched directly
    path: t.FilePath  #: path to a qgs project file
    rootLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to use as roots
    excludeLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to exclude
    flattenLayers: t.Optional[gws.base.layer.types.FlattenConfig]  #: flatten the layer hierarchy
    layerConfig: t.Optional[t.List[gws.base.layer.CustomConfig]]  #: custom configurations for specific layers


class Object(gws.base.layer.Group):
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

        layer_cfgs = gws.compact(self._layer(sl, depth=1) for sl in self.root_layers)
        if gws.is_empty(layer_cfgs):
            raise gws.Error(f'no source layers in {self.uid!r}')

        top_group = {
            'type': 'group',
            'title': '',
            'layers': layer_cfgs
        }

        top_cfg = gws.config.parser.parse(top_group, 'gws.ext.layer.group.Config')
        self.layers = gws.base.layer.add_layers_to_object(self, top_cfg.layers)

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

        if prov not in self.direct_render:
            la = self._qgis_based_layer(sl)
        elif prov == 'wms':
            la = self._wms_based_layer(sl)
        elif prov == 'wmts':
            la = self._wmts_based_layer(sl)
        else:
            gws.log.warn(f'directRender not supported for {prov!r}')
            la = self._qgis_based_layer(sl)

        if not la:
            return

        la['cache'] = self.var('cache')
        la['grid'] = self.var('grid')

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

    def _wms_based_layer(self, sl: t.SourceLayer):
        ds = sl.data_source

        layers = ds['layers']
        if not layers:
            return

        return {
            'type': 'wmsflat',
            'sourceLayers': {
                'names': ds['layers']
            },
            'url': self._make_wms_url(ds['url'], ds['params']),
            'getMapParams': ds['params'],
        }

    def _wmts_based_layer(self, sl: t.SourceLayer):
        ds = sl.data_source

        layers = ds['layers']
        if not layers:
            return

        opts = ds['options']

        return gws.compact({
            'type': 'wmts',
            'url': ds['url'].split('?')[0],
            'sourceLayer': ds['layers'][0],
            'sourceStyle': opts.get('styles'),
        })

    def _qgis_based_layer(self, sl: t.SourceLayer):
        return {
            'type': 'qgisflat',
            'sourceLayers': {
                'names': [sl.name]
            },
            'path': self.path,
        }

    def _flat_group_layer(self, sl: t.SourceLayer):
        if self.flatten.useGroups:
            names = [sl.name]
        else:
            ls = gws.gis.source.image_layers(sl)
            if not ls:
                return
            names = [s.name for s in ls]

        la = {
            'type': 'qgisflat',
            'sourceLayers': {
                'names': names
            },
            'path': self.path,
        }

        return la

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
        return gws.lib.net.add_params(url, p)
