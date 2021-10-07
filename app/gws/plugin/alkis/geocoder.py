"""ALKIS Geocoder action."""

import gws
import gws.types as t
import gws.base.api

from . import provider, util


@gws.ext.Config('action.alkisgeocoder')
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


@gws.ext.Object('action.alkisgeocoder')
class Object(gws.base.api.action.Object):
    provider: provider.Object

    def configure(self):
        self.provider = provider.create(self.root, self.config, shared=True)

    @gws.ext.command('api.alkisgeocoder.decode')
    def api_decode(self, req: gws.IWebRequest, p: GeocoderParams) -> GeocoderResponse:

        coords = []

        for ad in p.adressen:
            q = {k: ad.get(k) for k in _GEOCODER_ADDR_KEYS if ad.get(k)}

            if not q:
                coords.append(None)
                continue

            q['limit'] = 1

            res = self.provider.find_adresse(provider.FindAdresseQuery(q))

            if not res.total:
                coords.append(None)
                continue

            coords.append([
                round(res.features[0].shape.x, 2),
                round(res.features[0].shape.y, 2)
            ])

        return GeocoderResponse(coordinates=coords)
