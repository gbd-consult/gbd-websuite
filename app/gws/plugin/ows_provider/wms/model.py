"""Database-based models."""

import gws.base.model
import gws.base.feature
import gws.gis.source
import gws.gis.crs
import gws.gis.ows
import gws.gis.extent
import gws.types as t

from . import provider

gws.ext.new.model('wms')


class Config(gws.base.model.Config, provider.Config):
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """Source layers to search for."""


class Props(gws.base.model.Props):
    pass


class Object(gws.base.model.Object):
    provider: provider.Object
    sourceLayers: list[gws.SourceLayer]

    def configure(self):
        self.keyName = 'uid'
        self.geometryName = 'geometry'

        self.configure_provider()
        if not self.provider.get_operation(gws.OwsVerb.GetFeatureInfo):
            raise gws.Error(f'GetFeatureInfo is not supported for {self.provider.url!r}')

        self.configure_sources()

    def configure_provider(self):
        if self.cfg('_provider'):
            self.provider = self.cfg('_provider')
            return True
        self.provider = self.root.create_shared(provider.Object, self.config)
        return True

    def configure_sources(self):
        if self.cfg('_sourceLayers'):
            self.sourceLayers = self.cfg('_sourceLayers')
            return True
        self.sourceLayers = gws.gis.source.filter_layers(
            self.provider.sourceLayers,
            gws.gis.source.LayerFilter(self.cfg('sourceLayers'), isQueryable=True))
        if not self.sourceLayers:
            raise gws.Error(f'no queryable layers found in {self.provider.url!r}')

    def find_features(self, search, user, **kwargs):
        return [
            self.feature_from_data(fd, user, **kwargs)
            for fd in self.wms_search(search)
        ]

    def wms_search(self, search: gws.SearchArgs) -> list[gws.FeatureData]:
        v3 = self.provider.version >= '1.3'

        shape = search.shape
        if not shape or shape.type != gws.GeometryType.point:
            return []

        request_crs = self.provider.forceCrs
        if not request_crs:
            request_crs = gws.gis.crs.best_match(
                shape.crs,
                gws.gis.source.combined_crs_list(self.sourceLayers))

        box_size_m = 500
        box_size_deg = 1
        box_size_px = 500

        size = None

        if shape.crs.uom == gws.Uom.m:
            size = box_size_px * search.resolution
        if shape.crs.uom == gws.Uom.deg:
            # @TODO use search.resolution here as well
            size = box_size_deg
        if not size:
            gws.log.debug('cannot request crs {crs!r}, unsupported unit')
            return []

        bbox = (
            shape.x - (size / 2),
            shape.y - (size / 2),
            shape.x + (size / 2),
            shape.y + (size / 2),
        )

        bbox = gws.gis.extent.transform(bbox, shape.crs, request_crs)

        always_xy = self.provider.alwaysXY or not v3
        if request_crs.isYX and not always_xy:
            bbox = gws.gis.extent.swap_xy(bbox)

        layer_names = [sl.name for sl in self.sourceLayers]

        params = {
            'BBOX': bbox,
            'CRS' if v3 else 'SRS': request_crs.to_string(gws.CrsFormat.epsg),
            'WIDTH': box_size_px,
            'HEIGHT': box_size_px,
            'I' if v3 else 'X': box_size_px >> 1,
            'J' if v3 else 'Y': box_size_px >> 1,
            'LAYERS': layer_names,
            'QUERY_LAYERS': layer_names,
            'STYLES': [''] * len(layer_names),
            'VERSION': self.provider.version,
        }

        if search.limit:
            params['FEATURE_COUNT'] = search.limit
        if search.extraParams:
            params = gws.merge(params, gws.to_upper_dict(search.extraParams))

        op = self.provider.get_operation(gws.OwsVerb.GetFeatureInfo)
        if not op:
            return []

        if op.preferredFormat:
            params.setdefault('INFO_FORMAT', op.preferredFormat)

        args = self.provider.prepare_operation(op, params=params)
        text = gws.gis.ows.request.get_text(args)

        sfs = gws.gis.ows.featureinfo.parse(
            text,
            default_crs=request_crs,
            always_xy=always_xy)

        if sfs is None:
            gws.log.debug(f'WMS NOT_PARSED params={params!r}')
            return []
        gws.log.debug(f'WMS FOUND={len(sfs)} params={params!r}')

        for f in sfs:
            if f.shape:
                f.shape = f.shape.transformed_to(shape.crs)

        return sfs