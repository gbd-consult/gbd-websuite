import gws.web
import gws.types as t

import gws.ext.tool.alkis as alkis


class Config(t.WithType):
    """ALKIS Geocoder action."""
    pass


class GeocoderAddress(t.Data):
    gemeinde: str = ''
    gemarkung: str = ''
    strasse: str = ''
    hausnummer: str = ''


_GEOCODER_ADDR_KEYS = 'gemeinde', 'gemarkung', 'strasse', 'hausnummer'


class GeocoderParams(t.Data):
    adressen: t.List[GeocoderAddress]
    crs: t.Crs


class GeocoderResponse(t.Response):
    coordinates: t.List[t.Point]


class Object(gws.Object):
    def __init__(self):
        super().__init__()
        self.alkis: alkis.Object = None

    def configure(self):
        super().configure()
        self.alkis: alkis.Object = self.find_first('gws.ext.tool.alkis')
        if not self.alkis:
            raise ValueError('alkis tool not found')

    def api_decode(self, req: t.IRequest, p: GeocoderParams) -> GeocoderResponse:

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
