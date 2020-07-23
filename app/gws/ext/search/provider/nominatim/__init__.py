# http://wiki.openstreetmap.org/wiki/Nominatim

import gws
import gws.common.search.provider
import gws.common.template
import gws.gis.extent
import gws.gis.feature
import gws.gis.shape
import gws.tools.json2
import gws.tools.net

import gws.types as t

_NOMINATIM_CRS = gws.EPSG_4326
_NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search'

_DEFAULT_TEMPLATES = [
    t.Config(
        subject='feature.teaser',
        type='html',
        text='''
            <p class="head">{name | html}</p>
        '''
    ),
    t.Config(
        subject='feature.description',
        type='html',
        text='''
            <p class="head">{name | html}</p>
            <p class="text">{content | html}</p>
        '''
    ),
]


class Config(gws.common.search.provider.Config):
    """Nominatim (OSM) search provider"""

    country: t.Optional[str]  #: country to limit the search
    language: t.Optional[str]  #: language to return the results in


class Object(gws.common.search.provider.Object):
    def configure(self):
        super().configure()

        self.capabilties = gws.common.search.provider.CAPS_KEYWORD
        self.templates: t.List[t.ITemplate] = gws.common.template.bundle(self, self.var('templates'), _DEFAULT_TEMPLATES)

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        params = {
            'q': args.keyword,
            'addressdetails': 1,
            'polygon_geojson': 1,
            'format': 'json',
            'bounded': 1,
            'limit': args.limit,
        }

        if self.var('language'):
            params['accept-language'] = self.var('language')

        if self.var('country'):
            params['countrycodes'] = self.var('country')

        shape = self.context_shape(args)
        if not shape:
            return []

        params['viewbox'] = gws.gis.extent.transform(shape.extent, shape.crs, _NOMINATIM_CRS)

        features = []

        for rec in _query(params):
            uid = rec.get('place_id', ) or rec.get('osm_id')
            geom = rec.pop('geojson', None)

            if not geom:
                gws.log.debug(f'SKIP {uid}: no geometry')
                continue

            sh = gws.gis.shape.from_geometry(geom, _NOMINATIM_CRS).transformed_to(shape.crs)
            if not sh.intersects(shape):
                gws.log.debug(f'SKIP {uid}: no intersection')
                continue

            features.append(gws.gis.feature.Feature(
                uid=uid,
                attributes=_normalize(rec),
                shape=sh
            ))

        return features


def _query(params):
    try:
        res = gws.tools.net.http_request(_NOMINATIM_URL, params=params)
        return gws.tools.json2.from_string(res.text)
    except gws.tools.net.Error as e:
        gws.log.error('nominatim request error', e)
        return []


def _normalize(rec):
    # merge the address subrec into the main

    if 'address' in rec:
        for k, v in rec.pop('address').items():
            rec[k] = v

    # the problem is there's no fixed key for the 'name' (the key depends on the class)
    # however, display_name seems to always start with the 'name'

    dn = rec.get('display_name', '')
    rec['name'] = dn.split(',')[0].strip()
    rec['content'] = '' if rec['name'] == dn else dn

    return rec
