import gws
import gws.config
import gws.config.parser
import gws.gis.layer
import gws.gis.source
import gws.qgis
import gws.tools.net
import gws.types as t


class Config(gws.gis.layer.ProxiedConfig):
    """automatic QGIS layer"""

    directRender: t.Optional[t.List[str]]  #: QGIS providers that should be rendered directly
    directSearch: t.Optional[t.List[str]]  #: QGIS providers that should be searched directly
    path: t.filepath  #: path to a qgs project file


# NB: no 'props' for this layer, because it's just a group for the client


class Object(gws.gis.layer.Base):
    def __init__(self):
        super().__init__()
        self.path = ''
        self.direct_render = set()
        self.direct_search = set()
        self.layers: t.List[t.LayerObject] = []
        self.service: gws.qgis.Service = None
        self.path = ''

    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.service = gws.qgis.shared_service(self, self.config)

        self.direct_render = set(self.var('directRender', parent=True, default=[]))
        self.direct_search = set(self.var('directSearch', parent=True, default=[]))

        slf = self.var('sourceLayers')
        if slf:
            root_layers = gws.gis.source.filter_layers(self.service.layers, slf)
        else:
            root_layers = [sl for sl in self.service.layers if sl.a_level == 1]

        root_layers = gws.compact(self._layer(sl) for sl in root_layers)

        if not root_layers:
            raise gws.config.LoadError(f'no source layers in {self.uid!r}')

        top_group = {
            'type': 'group',
            'title': '',
            'layers': root_layers
        }

        top_cfg = gws.config.parser.parse(top_group, 'gws.ext.gis.layer.group.Config')
        for p in top_cfg.layers:
            try:
                self.layers.append(self.add_child('gws.ext.gis.layer', p))
            except Exception:
                gws.log.exception()

    @property
    def props(self):
        return gws.extend(super().props, {
            'type': 'group',
            'layers': self.layers,
        })

    def _layer(self, sl: t.SourceLayer):
        if sl.is_group:
            la = self._group_layer(sl)
        else:
            la = self._image_layer(sl)

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

        search = self._layer_search(sl)
        if search:
            la['search'] = search

        return la

    def _wms_based_layer(self, sl: t.SourceLayer):
        ds = sl.data_source

        layers = ds['layers']
        if not layers:
            return

        return {
            'type': 'box',
            'sourceLayers': {
                'names': ds['layers']
            },
            'source': {
                'type': 'wms',
                'url': ds['url'],
                'options': ds['options'],
                'params': ds['params'],
            }
        }

    def _wmts_based_layer(self, sl: t.SourceLayer):
        ds = sl.data_source

        layers = ds['layers']
        if not layers:
            return

        opts = ds['options']

        return {
            'type': 'tile',
            'source': {
                'type': 'wmts',
                'url': ds['url'].split('?')[0],
                'layer': ds['layers'][0],
                'format': opts.get('format', 'image/jpeg'),
                'style': opts.get('styles', 'default'),
            }
        }

    def _qgis_based_layer(self, sl: t.SourceLayer):
        ps = {
            'MAP': self.path,
            'LAYER': sl.name,
            'FORMAT': 'image/png',
            'REQUEST': 'GetLegendGraphic',
            'SERVICE': 'WMS',
            'STYLE': '',
            'VERSION': '1.1.1',
            'BOXSPACE': 0,
            'SYMBOLSPACE': 0,
            'LAYERTITLE': 'false',
            'LAYERFONTFAMILY': 'DejaVuSans',
            'ITEMFONTFAMILY': 'DejaVuSans',
        }

        legend = gws.tools.net.add_params(self.service.url, ps)

        return {
            'type': 'box',
            'sourceLayers': {
                'names': [sl.name]
            },
            'meta': sl.meta,
            'legend': legend,
            'source': {
                'type': 'qgis',
                'path': self.path,
            }

        }

    def _layer_search(self, sl: t.SourceLayer):
        ds = sl.data_source
        prov = ds['provider']

        if prov in self.direct_search:
            if prov == 'wms':
                return self._wms_direct_search(sl, ds)
            if prov == 'postgres':
                return self._postgres_direct_search(sl, ds)
            if prov == 'wfs':
                return self._wfs_direct_search(sl, ds)
            gws.log.warn(f'directSearch not supported for {prov!r}')

        # if no directSearch for this provider, the we make WMS request to the local QGIS
        # NB: cannot use type=wms here, because the qgis wms service is not started yet
        return {
            'providers': [
                {
                    'type': 'qgis_wms',
                    'path': self.path,
                    'layers': [sl.name],
                }
            ]
        }

    def _wms_direct_search(self, sl, ds):
        # @TODO: check if the remote layer is queriable
        return {
            'providers': [
                {
                    'type': 'wms',
                    'url': ds['url'],
                    'params': ds['params'],
                    'layers': ds['layers'],
                    # @TODO
                    # 'capsCacheMaxAge'
                }
            ]
        }

    def _postgres_direct_search(self, sl, ds):
        tab = sl.data_source['table']

        # 'table' can also be a select statement, in which case it might be enclosed in parens
        if tab.startswith('(') or tab.upper().startswith('SELECT '):
            return

        return {'providers': [{'type': 'qgis_postgis', 'ds': ds}]}

    def _wfs_direct_search(self, sl, ds):
        cfg = {
            'type': 'wfs',
            'url': ds['url'],
        }
        if gws.get(ds, 'typeName'):
            cfg['layers'] = [ds['typeName']]
        c = gws.get(ds, 'params.InvertAxisOrientation')
        if c == '1':
            cfg['axis'] = 'yx'

        return {
            'providers': [cfg]
        }

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
