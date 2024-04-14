"""Nominatim model."""

import gws
import gws.base.model
import gws.base.feature
import gws.base.shape
import gws.lib.net
import gws.lib.jsonx
import gws.gis.source
import gws.gis.crs
import gws.gis.bounds

import gws.types as t

gws.ext.new.model('nominatim')


class Config(gws.base.model.Config):
    """Nominatim model"""

    country: t.Optional[str]
    """country to limit the search"""
    language: t.Optional[str]
    """language to return the results in"""


class Object(gws.base.model.dynamic_model.Object):
    """Nominatim model."""

    country: str
    language: str

    serviceUrl = 'https://nominatim.openstreetmap.org/search'

    def configure(self):
        self.country = self.cfg('country')
        self.language = self.cfg('language')
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

    def find_features(self, search, user, **kwargs):
        params = {
            'q': search.keyword,
            'addressdetails': 1,
            'polygon_geojson': 1,
            'format': 'json',
            'bounded': 1,
            'dedupe': 1,
            'limit': search.limit,
        }

        if self.language:
            params['accept-language'] = self.cfg('language')

        if self.country:
            params['countrycodes'] = self.cfg('country')

        params['viewbox'] = gws.gis.bounds.transform(search.shape.bounds(), gws.gis.crs.WGS84).extent

        features = []

        for rec in self._query(params):
            uid = rec.get('place_id') or rec.get('osm_id')
            geom = rec.pop('geojson', None)

            if not geom:
                gws.log.debug(f'SKIP {uid}: no geometry')
                continue

            shape = gws.base.shape.from_geojson(geom, gws.gis.crs.WGS84, always_xy=True).transformed_to(search.shape.crs)
            if not shape.intersects(search.shape):
                gws.log.debug(f'SKIP {uid}: no intersection')
                continue

            features.append(self.feature_from_record(
                gws.FeatureRecord(uid=uid, shape=shape, attributes=self._normalize(rec)),
                user))

        return sorted(features, key=lambda f: (f.attr('name'), f.attr('osm_class'), f.attr('osm_type')))

    def _query(self, params) -> list[dict]:
        try:
            res = gws.lib.net.http_request(self.serviceUrl, params=params)
            return gws.lib.jsonx.from_string(res.text)
        except gws.lib.net.Error as e:
            gws.log.error('nominatim request error', e)
            return []

    def _normalize(self, rec):
        # merge the address subrec into the main

        if 'address' in rec:
            for k, v in rec.pop('address').items():
                rec['address_' + k] = v

        # ensure basic address fields

        rec['address_road'] = rec.get('address_road') or ''
        rec['address_building'] = rec.get('address_building') or rec.get('address_house_number') or ''
        rec['address_city'] = rec.get('address_city') or rec.get('address_town') or rec.get('address_village') or ''
        rec['address_country'] = rec.get('address_country') or ''

        # find out the "name"
        # the problem is there's no fixed key for the name (the key depends on the class)
        # however, display_name seems to always start with the 'name'

        dn = rec.get('display_name', '')
        rec['name'] = dn.split(',')[0].strip()

        # rename 'class' and 'type' for easier templating

        rec['osm_class'] = rec.pop('class', '')
        rec['osm_type'] = rec.pop('type', '')

        # remove geographic attributes

        for k in 'boundingbox', 'lat', 'lon':
            rec.pop(k, None)

        return rec
