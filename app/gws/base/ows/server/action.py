"""OWS server action."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.web
import gws.lib.mime
import gws.lib.xmlx

from . import core, layer_caps, error

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
        srv = cast(gws.OwsService, req.user.require(p.serviceUid, gws.ext.object.owsService))
        try:
            return srv.handle_request(req)
        except Exception as exc:
            err = error.from_exception(exc)
            verb = req.param('REQUEST')
            if verb in core.IMAGE_VERBS:
                return err.to_image_response()
            return err.to_xml_response('ows' if srv.isOwsCommon else 'ogc')

    @gws.ext.command.get('owsXml')
    def get_schema(self, req: gws.WebRequester, p: GetSchemaRequest) -> gws.ContentResponse:
        try:
            content = self._make_schema(req, p)
        except Exception as exc:
            return error.from_exception(exc).to_xml_response()
        return gws.ContentResponse(mime=gws.lib.mime.XML, content=content)

    def _make_schema(self, req, p) -> str:
        s = p.namespace
        if s.endswith('.xsd'):
            s = s[:-4]
        ns = gws.lib.xmlx.namespace.find_by_xmlns(s)
        if not ns:
            raise gws.NotFoundError(f'namespace not found: {p.namespace=}')

        lcs = []

        for p in self.root.find_all(gws.ext.object.layer):
            layer = cast(gws.Layer, p)
            if req.user.can_read(layer) and layer.ows.xmlNamespace and layer.ows.xmlNamespace.xmlns == ns.xmlns:
                lcs.append(layer_caps.for_layer(layer, req.user))

        xml = layer_caps.xml_schema(lcs, req.user)
        if not xml:
            raise gws.NotFoundError(f'cannot create schema: {p.namespace=}')

        return xml.to_string(
            with_xml_declaration=True,
            with_namespace_declarations=True,
            with_schema_locations=True
        )
