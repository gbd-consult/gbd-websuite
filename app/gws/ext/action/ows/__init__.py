"""OWS server."""

import gws
import gws.common.action
import gws.common.ows.service
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """OWS server action"""

    services: t.Optional[t.List[t.ext.ows.service.Config]]  #: services configuration


class Object(gws.common.action.Object):
    def __init__(self):
        super().__init__()
        self.services: t.List[t.IOwsService] = []

    def configure(self):
        super().configure()
        self.services = []

        for p in self.var('services', default=[]):
            self.services.append(self.add_child('gws.ext.ows.service', p))

    def http_get_service(self, req: t.IRequest, _) -> t.HttpResponse:
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
        uid = req.param('uid')
        for service in self.services:
            if service.uid == uid:
                return service
