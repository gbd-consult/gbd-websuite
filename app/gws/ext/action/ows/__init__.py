import gws
import gws.web.error
import gws.types as t

from . import wms


class Config(t.WithTypeAndAccess):
    """OWS server action"""
    pass


class Object(gws.Object):

    def http_get(self, req, p) -> t.HttpResponse:
        ps = {k.lower(): v for k, v in req.params.items()}
        service = ps.get('service').lower()
        if service == 'wms':
            return wms.request(self, req, ps)
        raise gws.web.error.NotFound()

