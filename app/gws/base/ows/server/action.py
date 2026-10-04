"""OWS server action."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.web
import gws.lib.mime
import gws.lib.xmlx

from . import core, layer_caps, error


class GetServiceRequest(gws.Request):
    """Request to an OWS service."""

    serviceUid: str
    """Service uid."""


class GetSchemaRequest(gws.Request):
    """Request for an XML schema."""

    namespace: str
    """Namespace prefix, optionally with the ``.xsd`` extension."""


@gws.ext.config.action('ows')
class Config(gws.base.action.Config):
    """Action that serves the configured OWS services."""


@gws.ext.object.action('ows')
class Object(gws.base.action.Object):
    """OWS action.

    Dispatches requests to the configured OWS services and serves XML schemas
    generated for layer namespaces.
    """

    @gws.ext.command.get('owsService')
    def get_service(self, req: gws.WebRequester, p: GetServiceRequest) -> gws.ContentResponse:
        """Handle an OWS service request."""

        return self._handle_service(req, p)

    @gws.ext.command.post('owsService')
    def post_service(self, req: gws.WebRequester, p: GetServiceRequest) -> gws.ContentResponse:
        """Handle an OWS service request."""

        return self._handle_service(req, p)

    def _handle_service(self, req: gws.WebRequester, p: GetServiceRequest) -> gws.ContentResponse:
        """Locate the service, check access and pass the request to it."""
        srv = cast(gws.OwsService, self.root.get(p.serviceUid, gws.ext.object.owsService))
        if not srv:
            raise gws.NotFoundError(f'{p.serviceUid=} not found')
        if not req.user.can_use(srv):
            raise gws.ForbiddenError(f'{p.serviceUid=} forbidden')
        return srv.handle_request(req)

    @gws.ext.command.get('owsXml')
    def get_schema(self, req: gws.WebRequester, p: GetSchemaRequest) -> gws.ContentResponse:
        """Return an XML schema for the layers with the given namespace prefix."""

        try:
            content = self._make_schema(req, p)
        except Exception as exc:
            return error.from_exception(exc).to_xml_response()
        return gws.ContentResponse(mimeType=gws.lib.mime.XML, content=content)

    def _make_schema(self, req, p) -> str:
        """Create the XML schema document for a namespace."""
        s = p.namespace
        if s.endswith('.xsd'):
            s = s[:-4]
        lcs = []

        for la in self.root.find_all(gws.ext.object.layer):
            layer = cast(gws.Layer, la)
            if req.user.can_read(layer) and layer.ows.xmlNamespace and layer.ows.xmlNamespace.prefix == s:
                lcs.append(layer_caps.for_layer(layer, req.user))

        if not lcs:
            raise gws.NotFoundError(f'namespace not found: {p.namespace=}')

        el, opts = layer_caps.xml_schema(lcs, req.user)
        if not el:
            raise gws.NotFoundError(f'cannot create schema: {p.namespace=}')

        opts.withNamespaceDeclarations = True
        opts.withSchemaLocations = True
        opts.withXmlDeclaration = True
        
        return el.to_string(opts)
