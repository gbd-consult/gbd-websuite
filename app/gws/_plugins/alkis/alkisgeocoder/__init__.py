"""ALKIS Geocoder action."""

import gws.ext.helper.alkis as alkis

import gws
import gws.types as t
import gws.base.api


class Config(gws.WithAccess):
    """ALKIS Geocoder action."""
    pass


class GeocoderAddress(gws.Data):
    gemeinde: str = ''
    gemarkung: str = ''
    strasse: str = ''
    hausnummer: str = ''


_GEOCODER_ADDR_KEYS = 'gemeinde', 'gemarkung', 'strasse', 'hausnummer'


class GeocoderParams(gws.Data):
    adressen: t.List[GeocoderAddress]
    crs: gws.Crs


class GeocoderResponse(gws.Response):
    coordinates: t.List[gws.Point]


class Object(gws.base.api.Action):
    alkis: alkis.Object

    def configure(self):
        
        self.alkis = t.cast(alkis.Object, self.root.find_first('gws.ext.helper.alkis'))
        if not self.alkis:
            raise ValueError('alkis helper not found')

    def api_decode(self, req: gws.IWebRequest, p: GeocoderParams) -> GeocoderResponse:

        coords = []

        for ad in p.adressen:
            q = {k: ad.get(k) for k in _GEOCODER_ADDR_KEYS if ad.get(k)}

            if not q:
                coords.append(None)
                continue

            q['limit'] = 1

            res = self.alkis.find_adresse(alkis.FindAdresseQuery(q))

            if not res.total:
                coords.append(None)
                continue

            coords.append([
                round(res.features[0].shape.x, 2),
                round(res.features[0].shape.y, 2)
            ])

        return GeocoderResponse(coordinates=coords)
