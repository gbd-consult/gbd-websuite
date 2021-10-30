"""ALKIS Geocoder action."""

import gws
import gws.base.api
import gws.types as t

from . import provider as provider_module, types


@gws.ext.Config('action.alkisgeocoder')
class Config(provider_module.Config):
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
    provider: provider_module.Object

    def configure(self):
        self.provider = provider_module.create(self.root, self.config, shared=True)

    @gws.ext.command('api.alkisgeocoder.decode')
    def api_decode(self, req: gws.IWebRequest, p: GeocoderParams) -> GeocoderResponse:
        return GeocoderResponse(
            coordinates=[
                self.coords(adresse)
                for adresse in p.adressen
            ])

    def coords(self, adresse):
        q = {k: adresse.get(k) for k in _GEOCODER_ADDR_KEYS if adresse.get(k)}

        if not q:
            return

        q['limit'] = 1

        res = self.provider.find_adresse(types.FindAdresseQuery(q))

        if not res.total:
            return

        for feature in res.features:
            if not feature.shape:
                gws.log.warn(f'feature {feature.uid!r} has no shape')
                continue
            return (
                round(feature.shape.x, 2),
                round(feature.shape.y, 2)
            )
