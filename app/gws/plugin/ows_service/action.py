"""OWS server."""

import gws
import gws.base.api
import gws.base.web.error
import gws.types as t


class ServiceParams(gws.Params):
    serviceUid: str


@gws.ext.Config('action.ows')
class Config(gws.base.api.action.Config):
    """OWS server action"""

    services: t.List[gws.ext.ows.service.Config]  #: services configuration


@gws.ext.Object('action.ows')
class Object(gws.base.api.action.Object):
    services: t.List[gws.IOwsService]

    def configure(self):
        self.services = self.create_children('gws.ext.ows.service', self.var('services'))

    @gws.ext.command('get.ows.service')
    def service(self, req: gws.IWebRequest, p: ServiceParams) -> gws.ContentResponse:
        srv = self._find_service(p.serviceUid)
        if not srv:
            gws.log.debug(f'service not found uid={p.serviceUid!r}')
            raise gws.base.web.error.NotFound()

        gws.log.debug(f'found service={srv.uid}')

        if not req.user.can_use(srv):
            return srv.error_response(gws.base.web.error.Forbidden())

        try:
            return srv.handle_request(req)
        except gws.base.web.error.HTTPException as err:
            return srv.error_response(err)
        except:
            gws.log.exception()
            return srv.error_response(gws.base.web.error.InternalServerError())

    def _find_service(self, service_uid) -> gws.IOwsService:
        for srv in self.services:
            if srv.uid == service_uid:
                return srv
