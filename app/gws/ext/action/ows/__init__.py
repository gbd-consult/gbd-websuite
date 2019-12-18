import os

import gws
import gws.common.ows.service
import gws.web
import gws.types as t


class Config(t.WithTypeAndAccess):
    """OWS server action"""

    services: t.Optional[t.List[t.ext.ows.service.Config]]  #: services configuration


class Object(gws.ActionObject):
    @property
    def props(self):
        # no client props for this action
        return None

    def __init__(self):
        super().__init__()
        self.services: t.List[gws.common.ows.service.Object] = []

    def configure(self):
        super().configure()
        self.services = []

        for p in self.var('services', default=[]):
            self.services.append(self.add_child('gws.ext.ows.service', p))

    def http(self, req: gws.web.AuthRequest, _) -> t.HttpResponse:
        gws.p(req.params)
        gws.p(req.post_data)

        service = self._find_service(req)
        if not service:
            raise gws.web.error.NotFound()

        gws.log.debug(f'found service={service.uid}')

        if not req.user.can_use(service):
            return service.error_response(403)

        try:
            return service.handle(req)
        except gws.web.error.HTTPException as err:
            return service.error_response(err.code)
        except:
            gws.log.exception()
            return service.error_response(500)

    def _find_service(self, req):
        s = req.param('serviceName')
        for service in self.services:
            if service.name == s:
                return service
