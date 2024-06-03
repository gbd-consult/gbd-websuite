"""OWS server."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.web
import gws.lib.mime
import gws.lib.xmlx

from . import layer_caps

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
        try:
            return self._get_service(req, p)
        except Exception as exc:
            return self._xml_error(exc)

    def _get_service(self, req, p):
        srv = cast(gws.OwsService, req.user.require(p.serviceUid, gws.ext.object.owsService))
        return srv.handle_request(req)

    @gws.ext.command.get('owsXmlSchema')
    def get_schema(self, req: gws.WebRequester, p: GetSchemaRequest) -> gws.ContentResponse:
        try:
            return self._get_schema(req, p)
        except Exception as exc:
            return self._xml_error(exc)

    def _get_schema(self, req, p):
        ns = gws.lib.xmlx.namespace.find_by_xmlns(p.namespace)
        if not ns:
            raise gws.NotFoundError(f'namespace not found: {p.namespace=}')

        lcs = []

        for p in self.root.find_all(gws.ext.object.layer):
            la = cast(gws.Layer, p)
            if req.user.can_read(la) and la.owsOptions.xmlNamespace and la.owsOptions.xmlNamespace.xmlns == ns.xmlns:
                lcs.append(layer_caps.for_layer(la, req.user))

        xml = layer_caps.xml_schema(lcs, req.user)
        if not xml:
            raise gws.NotFoundError(f'cannot create schema: {p.namespace=}')

        return gws.ContentResponse(
            mime=gws.lib.mime.XML,
            content=xml.to_string(with_xml_declaration=True, with_namespace_declarations=True, with_schema_locations=True)
        )

    def _xml_error(self, exc: Exception):
        web_exc = gws.base.web.error.from_exception(exc)
        xml = gws.lib.xmlx.tag('ServiceExceptionReport/ServiceException', {'code': web_exc.code}, web_exc.description)

        # @TODO status, check OGC 17-007r1

        return gws.ContentResponse(
            mime=gws.lib.mime.XML,
            content=xml.to_string(with_xml_declaration=True),
            status=200,
        )
