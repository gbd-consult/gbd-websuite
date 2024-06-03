"""OWS server."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.web
import gws.lib.mime
import gws.lib.xmlx

from . import layer_caps, error

gws.ext.new.action('ows')


class GetServiceRequest(gws.Request):
    serviceUid: str


class GetSchemaRequest(gws.Request):
    namespace: str


class Config(gws.base.action.Config):
    """OWS server action"""


class Object(gws.base.action.Object):
    image_verbs = {
        gws.OwsVerb.GetMap,
        gws.OwsVerb.GetTile,
        gws.OwsVerb.GetLegendGraphic,
    }

    @gws.ext.command.get('owsService')
    def get_service(self, req: gws.WebRequester, p: GetServiceRequest) -> gws.ContentResponse:
        try:
            return self._get_service(req, p)
        except Exception as exc:
            err = error.from_exception(exc)
            verb = req.param('REQUEST')
            if verb in self.image_verbs:
                return err.to_image()
            return err.to_xml()

    def _get_service(self, req, p):
        srv = cast(gws.OwsService, req.user.require(p.serviceUid, gws.ext.object.owsService))
        return srv.handle_request(req)

    @gws.ext.command.get('owsXmlSchema')
    def get_schema(self, req: gws.WebRequester, p: GetSchemaRequest) -> gws.ContentResponse:
        try:
            return self._get_schema(req, p)
        except Exception as exc:
            return error.from_exception(exc).to_xml()

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
