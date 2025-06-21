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
        return self._handle_service(req, p)

    @gws.ext.command.post('owsService')
    def post_service(self, req: gws.WebRequester, p: GetServiceRequest) -> gws.ContentResponse:
        return self._handle_service(req, p)

    def _handle_service(self, req: gws.WebRequester, p: GetServiceRequest) -> gws.ContentResponse:
        srv = cast(gws.OwsService, self.root.get(p.serviceUid, gws.ext.object.owsService))
        if not srv:
            raise gws.NotFoundError(f'{p.serviceUid=} not found')
        if not req.user.can_use(srv):
            raise gws.ForbiddenError(f'{p.serviceUid=} forbidden')
        return srv.handle_request(req)

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
        ns = gws.lib.xmlx.namespace.get(s)
        if not ns:
            raise gws.NotFoundError(f'namespace not found: {p.namespace=}')

        lcs = []

        for la in self.root.find_all(gws.ext.object.layer):
            layer = cast(gws.Layer, la)
            if req.user.can_read(layer) and layer.ows.xmlNamespace and layer.ows.xmlNamespace.xmlns == ns.xmlns:
                lcs.append(layer_caps.for_layer(layer, req.user))

        el, opts = layer_caps.xml_schema(lcs, req.user)
        if not el:
            raise gws.NotFoundError(f'cannot create schema: {p.namespace=}')

        opts.withNamespaceDeclarations = True
        opts.withSchemaLocations = True
        opts.withXmlDeclaration = True
        
        return el.to_string(opts)
