import gws
import gws.config
import gws.config.parser
import gws.gis.layer
import gws.gis.source
import gws.qgis
import gws.tools.net
import gws.types as t
import gws.ows.util


class Config(gws.gis.layer.ProxiedConfig):
    """automatic QGIS layer"""

    directRender: t.Optional[t.List[str]]  #: QGIS providers that should be rendered directly
    directSearch: t.Optional[t.List[str]]  #: QGIS providers that should be searched directly
    path: t.filepath  #: path to a qgs project file
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use


class Object(gws.gis.layer.Base):
    def __init__(self):
        super().__init__()
        self.path = ''
        self.direct_render = set()
        self.direct_search = set()
        self.source_layers: t.List[t.SourceLayer] = []
        self.layers: t.List[t.LayerObject] = []
        self.service: gws.qgis.Service = None
        self.path = ''

    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.service = gws.qgis.shared_service(self, self.config)

        self.direct_render = set(self.var('directRender', default=[]))
        self.direct_search = set(self.var('directSearch', default=[]))

        slf = self.var('sourceLayers') or t.Data({'names': [], 'pattern': '', 'level': 1})
        self.source_layers = gws.gis.source.filter_layers(self.service.layers, slf)

        layer_cfgs = gws.compact(self._layer(sl) for sl in self.source_layers)
        if gws.is_empty(layer_cfgs):
            raise gws.Error(f'no source layers in {self.uid!r}')

        top_group = {
            'type': 'group',
            'title': '',
            'layers': layer_cfgs
        }

        top_cfg = gws.config.parser.parse(top_group, 'gws.ext.layer.group.Config')
        self.layers = gws.gis.layer.add_layers_to_object(self, top_cfg.layers)

    def render_legend(self):
        return self.service.get_legend(sl.name for sl in self.source_layers)

    @property
    def props(self):
        return gws.extend(super().props, {
            'type': 'group',
            'layers': self.layers,
        })

    def _layer(self, sl: t.SourceLayer):
        la = self._group_layer(sl) if sl.is_group else self._image_layer(sl)

        if not la:
            return

        la = gws.extend(la, {
            'uid': sl.name,
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
            'type': 'wms',
            'sourceLayers': {
                'names': ds['layers']
            },
            'url': ds['url'],
            'params': ds['params'],
        }

    def _wmts_based_layer(self, sl: t.SourceLayer):
        ds = sl.data_source

        layers = ds['layers']
        if not layers:
            return

        opts = ds['options']

        return {
            'type': 'wmts',
            'url': ds['url'].split('?')[0],
            'layer': ds['layers'][0],
            'format': opts.get('format', 'image/jpeg'),
            'style': opts.get('styles', 'default'),
        }

    def _qgis_based_layer(self, sl: t.SourceLayer):
        return {
            'type': 'qgiswms',
            'sourceLayers': {
                'names': [sl.name]
            },
            'path': self.path,
        }

    def _layer_search_provider(self, sl: t.SourceLayer):
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

        # if no directSearch for this provider, the we make WMS request to the local QGIS
        # NB: cannot use type=wms here, because the qgis wms service is not started yet
        return {
            'type': 'qgiswms',
            'path': self.path,
            'layers': [sl.name],
        }

    def _wms_search_provider(self, sl, ds):
        # @TODO: check if the remote layer is queriable
        return {
            'type': 'wms',
            'url': ds['url'],
            'params': ds['params'],
            'layers': ds['layers'],
            # @TODO
            # 'capsCacheMaxAge'
        }

    def _postgres_search_provider(self, sl, ds):
        tab = sl.data_source['table']

        # 'table' can also be a select statement, in which case it might be enclosed in parens
        if tab.startswith('(') or tab.upper().startswith('SELECT '):
            return

        return {'type': 'qgispostgres', 'ds': ds}

    def _wfs_search_provider(self, sl, ds):
        cfg = {
            'type': 'wfs',
            'url': ds['url'],
        }
        if gws.get(ds, 'typeName'):
            cfg['layers'] = [ds['typeName']]
        c = gws.get(ds, 'params.InvertAxisOrientation')
        if c == '1':
            cfg['axis'] = 'yx'

        return cfg

    def _group_layer(self, sl: t.SourceLayer):
        layers = gws.compact(self._layer(la) for la in sl.layers)
        if not layers:
            return
        return {
            'type': 'group',
            'title': sl.title,
            'uid': sl.name,
            'layers': layers
        }
