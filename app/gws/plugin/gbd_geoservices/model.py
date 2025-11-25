"""GBD Geoservices model."""

import re

import gws
import gws.base.feature
import gws.base.model
import gws.base.shape
import gws.lib.bounds
import gws.lib.crs
import gws.gis.source
import gws.lib.jsonx
import gws.lib.net

gws.ext.new.model('gbd_geoservices')


class Config(gws.base.model.Config):
    """GBD Geoservices model."""

    apiKey: str
    """API key for GBD Geoservices."""


class Object(gws.base.model.default_model.Object):
    """GBD Geoservices model."""

    apiKey: str

    serviceUrl = 'https://geoservices.gbd-consult.de/search'

    def configure(self):
        self.apiKey = self.cfg('apiKey')
        self.uidName = 'uid'
        self.geometryName = 'geometry'
        self.loadingStrategy = gws.FeatureLoadingStrategy.all

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            canCreate=False,
            canDelete=False,
            canWrite=False,
        )

    def find_features(self, search, mc, **kwargs):
        request = {'page': 0, 'tags': {}}

        if search.shape:
            geometry_tolerance = 0.0

            if search.tolerance:
                n, u = search.tolerance
                geometry_tolerance = n * (search.resolution or 1) if u == 'px' else n

            search_shape = search.shape.tolerance_polygon(geometry_tolerance)
            request['viewbox'] = gws.lib.bounds.wgs_extent(search_shape.bounds())

        kw = search.keyword or ''
        if kw:
            request['intersect'] = 1
            # crude heuristics to check if this is an "address" or a "name"
            if re.search(r'\s\d', kw):
                request['address'] = kw
            else:
                request['name'] = kw

        features = {}

        res = self._query(request)

        for f in res['results']['features']:
            a = {k.replace(':', '_'): v for k, v in sorted(f['properties'].items())}
            shape = gws.base.shape.from_geojson(f['geometry'], gws.lib.crs.WGS84, always_xy=True)

            if search.shape and search.shape.type == gws.GeometryType.polygon and not shape.intersects(search.shape):
                continue

            address = ' '.join(
                [
                    a.get('addr_street') or '',
                    a.get('addr_housenumber') or '',
                    a.get('addr_postcode') or '',
                    a.get('addr_city') or '',
                ]
            )

            a['address'] = ' '.join(address.split())
            a['name'] = a.get('name') or ''

            if a['name'] and a['address']:
                a['title'] = a['name'] + ' (' + a['address'] + ')'
            else:
                a['title'] = a['name'] or a['address']

            rec = gws.FeatureRecord(uid=f['id'], attributes=a, shape=shape)
            features[a['title']] = self.feature_from_record(rec, mc)

        return [f for _, f in sorted(features.items())]

    def _query(self, request) -> dict:
        try:
            res = gws.lib.net.http_request(
                self.serviceUrl,
                method='POST',
                headers={'x-api-key': self.apiKey},
                json=request,
            )
            return gws.lib.jsonx.from_string(res.text)
        except gws.lib.net.Error as e:
            gws.log.error('geoservices request error', e)
            return {}
