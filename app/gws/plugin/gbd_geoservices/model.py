"""GBD Geoservices model."""

import gws
import gws.base.feature
import gws.base.model
import gws.base.shape
import gws.lib.bounds
import gws.lib.crs
import gws.gis.source
import gws.lib.jsonx
import gws.lib.net


@gws.ext.config.model('gbd_geoservices')
class Config(gws.base.model.Config):
    """Read-only model for features from GBD Geoservices."""

    apiKey: str
    """API key for GBD Geoservices."""


@gws.ext.object.model('gbd_geoservices')
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
        kw = (search.keyword or '').strip()

        tolerance = 0.0
        if search.tolerance:
            n, u = search.tolerance
            tolerance = n * (search.resolution or 1) if u == 'px' else n

        area = None
        if search.shape:
            area = search.shape.tolerance_polygon(tolerance).transformed_to(gws.lib.crs.WGS84)

        request = {}

        if kw:
            request['mode'] = 'phrase'
            request['text'] = kw
            if area:
                request['area'] = area.to_wkt(output_dimension=2)
                request['within'] = False
        elif area:
            request['mode'] = 'point'
            if tolerance > 0:
                request['radius'] = min(tolerance, _MAX_POINT_RADIUS)
        else:
            return []

        request['limit'] = min(search.limit or _MAX_LIMIT, _MAX_LIMIT)
        if area:
            c = area.centroid()
            request['near'] = [c.geom.x, c.geom.y]

        res = self._request(self.serviceUrl, method='POST', json=request)
        if not res:
            return []

        results = res.get('results') or {}
        features = results.get('features') or []

        out = []

        for f in features:
            a = self._attributes(f['properties'])
            if not a:
                gws.log.warning(f'geoservices: skipping feature without name, address or category: {f.get("id")}')
                continue
            shape = gws.base.shape.from_geojson(f['geometry'], gws.lib.crs.WGS84, always_xy=True)
            rec = gws.FeatureRecord(uid=f['id'], attributes=a, shape=shape)
            gws.log.debug(f'geoservices: feature record: {rec=}')
            out.append(self.feature_from_record(rec, mc))

        return out

    def _request(self, url, **kwargs) -> dict:
        res = gws.lib.net.http_request(
            url,
            headers={'Authorization': f'Bearer {self.apiKey}'},
            **kwargs,
        )
        if not res.ok:
            gws.log.error(f'geoservices request error: {res.status_code} {res.text}')
            return {}
        gws.log.debug(f'{res.text=}')
        return gws.lib.jsonx.from_string(res.text)

    def _attributes(self, props: dict) -> dict | None:
        name = props.get('name') or ''

        addr1 = _join([props.get('address_street'), props.get('address_housenumber')])
        addr2 = _join([props.get('address_postcode'), props.get('address_city')])

        category = props.get('category_name') or ''
        icon = props.get('icon') or ''
        places = [p['name'] for p in props.get('places') or []]
        places_line = ' • '.join(places)
        place = places[-1] if places else ''

        a = dict(title='', subtitle='', teaser='', details='', icon=icon)

        if name:
            a['title'] = name
            a['subtitle'] = category
            a['teaser'] = places_line
            if addr1:
                a['details'] = addr1 + '\n' + addr2
            else:
                a['details'] = places_line
        elif addr1:
            a['title'] = addr1 + '\n' + addr2
            a['subtitle'] = category
        elif category:
            a['title'] = category
            a['subtitle'] = ''
            a['teaser'] = place
            if addr1:
                a['details'] = addr1 + '\n' + addr2
            else:
                a['details'] = places_line
        else:
            return None

        return a


_MAX_LIMIT = 100
_MAX_POINT_RADIUS = 10_000


def _join(parts):
    return ' '.join(' '.join(str(p) for p in parts if p).split())
