"""OWS server."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.web
import gws.lib.mime
import gws.lib.xmlx

from . import util

gws.ext.new.action('ows')


class GetServiceRequest(gws.Request):
    serviceUid: str


class GetSchemaRequest(gws.Request):
    namespace: str


class Config(gws.base.action.Config):
    """OWS server action"""


class Object(gws.base.action.Object):
    @gws.ext.command.get('owsService')
    def get_service(self, req: gws.WebRequester, p: GetServiceRequest) -> gws.ContentResponse:

        srv = self._find_service(req, p)
        if not srv:
            raise gws.NotFoundError(f'service not found: {p.serviceUid!r}')

        if not req.user.can_use(srv):
            return self._xml_error(gws.base.web.error.Forbidden())

        try:
            return srv.handle_request(req)
        except gws.base.web.error.HTTPException as exc:
            # @TODO image errors
            gws.log.exception()
            gws.log.error(f'OWS error={exc!r}')
            return self._xml_error(exc)
        except Exception:
            gws.log.exception()
            return self._xml_error(gws.base.web.error.InternalServerError())

    @gws.ext.command.get('owsXmlSchema')
    def get_schema(self, req: gws.WebRequester, p: GetSchemaRequest) -> gws.ContentResponse:
        ns = gws.lib.xmlx.namespace.find_by_xmlns(p.namespace)
        if not ns:
            raise gws.NotFoundError(f'namespace not found: {p.namespace=}')

        lcs = []

        for p in self.root.find_all(gws.ext.object.layer):
            la = cast(gws.Layer, p)
            if req.user.can_read(la) and la.owsOptions.xmlNamespace and la.owsOptions.xmlNamespace.xmlns == ns.xmlns:
                lcs.append(util.layer_caps_for_layer(la, req.user))

        xml = util.xml_schema(lcs)
        if not xml:
            raise gws.NotFoundError(f'cannot create schema')

        return gws.ContentResponse(
            mime=gws.lib.mime.XML,
            content=xml.to_string(with_xml_declaration=True, with_namespace_declarations=True, with_schema_locations=True)
        )

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

        xml = gws.lib.xmlx.tag('ServiceExceptionReport/ServiceException', {'code': status}, description)

        # @TODO status, check OGC 17-007r1

        return gws.ContentResponse(
            mime=gws.lib.mime.XML,
            content=xml.to_string(with_xml_declaration=True),
            status=200,
        )
