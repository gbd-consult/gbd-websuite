"""OWS server."""

import gws
import gws.types as t
import gws.base.api
import gws.base.ows.service
import gws.base.web.error


class Config(gws.WithAccess):
    """OWS server action"""

    services: t.Optional[t.List[gws.ext.ows.service.Config]]  #: services configuration


class Object(gws.base.api.Action):
    def configure(self):
        

        self.services: t.List[gws.IOwsService] = []
        for p in self.var('services', default=[]):
            self.services.append(t.cast(gws.IOwsService, self.create_child('gws.ext.ows.service', p)))

    def http_service(self, req: gws.IWebRequest, _) -> gws.ContentResponse:
        service = self._find_service(req)
        if not service:
            gws.log.debug('service not found')
            raise gws.base.web.error.NotFound()

        gws.log.debug(f'found service={service.uid}')

        if not req.user.can_use(service):
            return service.error_response(gws.base.web.error.Forbidden())

        try:
            return service.handle(req)
        except gws.base.web.error.HTTPException as err:
            return service.error_response(err)
        except:
            gws.log.exception()
            return service.error_response(gws.base.web.error.InternalServerError())

    def _find_service(self, req):
        uid = req.param('uid')
        for service in self.services:
            if service.uid == uid:
                return service
