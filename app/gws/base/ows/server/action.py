"""OWS server."""

from typing import Optional

import gws
import gws.base.action
import gws.base.web
import gws.lib.mime
import gws.lib.xmlx as xmlx


gws.ext.new.action('ows')


class ServiceRequest(gws.Request):
    serviceUid: str


class Config(gws.base.action.Config):
    """OWS server action"""


class Object(gws.base.action.Object):
    @gws.ext.command.get('owsService')
    def service(self, req: gws.WebRequester, p: ServiceRequest) -> gws.ContentResponse:

        srv = self._find_service(req, p)
        if not srv:
            raise gws.NotFoundError(f'service not found: {p.serviceUid!r}')

        if not req.user.can_use(srv):
            return self._xml_error(gws.base.web.error.Forbidden())

        try:
            return srv.handle_request(req)
        except gws.base.web.error.HTTPException as exc:
            # @TODO image errors
            gws.log.error(f'OWS error={exc!r}')
            return self._xml_error(exc)
        except Exception:
            gws.log.exception()
            return self._xml_error(gws.base.web.error.InternalServerError())

    def _find_service(self, req, p) -> Optional[gws.OwsService]:
        if p.projectUid:
            project = req.user.require_project(p.projectUid)
            for s in project.owsServices:
                if s.uid == p.serviceUid:
                    return s

        for s in self.root.app.owsServices:
            if s.uid == p.serviceUid:
                return s

    def _xml_error(self, exc: Exception):
        try:
            status = int(gws.u.get(exc, 'code', 500))
        except Exception:
            status = 500

        description = gws.u.get(exc, 'description') or f'Error {status}'

        xml = gws.lib.xmlx.tag(
            'ServiceExceptionReport',
            ('ServiceException', {'code': status}, description)
        )

        # @TODO status, check OGC 17-007r1

        return gws.ContentResponse(mime=gws.lib.mime.XML, content=xml.to_string(with_xml_declaration=True), status=200)
