"""OWS Service."""

from typing import Optional, Callable, cast

import gws
import gws.base.layer.core
import gws.base.template
import gws.base.web
import gws.config.util
import gws.gis.bounds
import gws.gis.crs
import gws.gis.extent
import gws.gis.gml
import gws.gis.render
import gws.gis.source
import gws.lib.date
import gws.lib.image
import gws.lib.metadata
import gws.lib.mime

from . import request, error


class Config(gws.ConfigWithAccess):
    extent: Optional[gws.Extent]
    """Service extent."""
    metadata: Optional[gws.Metadata]
    """Service metadata."""
    rootLayerUid: str = ''
    """Root layer uid."""
    maxFeatureCount: int = 10000
    """Max number of features per page."""
    defaultFeatureCount: int = 1000
    """Default number of features per page."""
    searchTolerance: gws.UomValueStr = '10px'
    """Search pixel tolerance."""
    supportedCrs: Optional[list[gws.CrsName]]
    """Supported CRS for this service."""
    templates: Optional[list[gws.ext.config.template]]
    """Service XML templates."""
    imageFormats: Optional[list[str]]
    """Supported image formats."""
    imageOptions: Optional[dict]
    """Image options."""
    updateSequence: Optional[str]
    """Service update sequence."""
    withInspireMeta: bool = False
    """Use INSPIRE Metadata."""
    withStrictParams: bool = False
    """Use strict params checking."""


class Object(gws.OwsService):
    """Baseclass for OWS services."""

    handlers: dict[gws.OwsVerb, Callable]

    def configure(self):
        self.project = self.find_closest(gws.ext.object.project)

        self.updateSequence = self.cfg('updateSequence')
        self.withInspireMeta = self.cfg('withInspireMeta')
        self.withStrictParams = self.cfg('withStrictParams')

        self.maxFeatureCount = self.cfg('maxFeatureCount')
        self.defaultFeatureCount = self.cfg('defaultFeatureCount')
        self.searchTolerance = self.cfg('searchTolerance')

        self.imageFormats = self.cfg('imageFormats') or [gws.lib.mime.PNG, gws.lib.mime.JPEG]
        self.imageOptions = self.cfg('imageOptions') or {}

        self.configure_bounds()
        self.configure_templates()
        self.configure_operations()
        self.configure_metadata()

        self.handlers = {}

    def configure_bounds(self):
        crs_list = [gws.gis.crs.require(s) for s in self.cfg('supportedCrs', default=[])]
        if not crs_list:
            crs_list = [self.project.map.bounds.crs] if self.project else [gws.gis.crs.WEBMERCATOR]

        p = self.cfg('extent')
        if p:
            bounds = gws.Bounds(crs=crs_list[0], extent=gws.gis.extent.from_list(p))
        elif self.project:
            bounds = self.project.map.bounds
        else:
            bounds = gws.Bounds(crs=crs_list[0], extent=crs_list[0].extent)

        self.supportedBounds = [gws.gis.bounds.transform(bounds, crs) for crs in crs_list]
        return True

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self)

    def configure_operations(self):
        pass

    def template_formats(self, verb: gws.OwsVerb):
        fs = []
        for tpl in self.templates:
            if tpl.subject == f'ows.{verb}':
                fs.extend(tpl.mimeTypes)
        return sorted(set(fs))

    def configure_metadata(self):
        self.metadata = gws.lib.metadata.merge(
            gws.Metadata(),
            self.project.metadata if self.project else None,
            self.root.app.metadata,
            gws.lib.metadata.from_config(self.cfg('metadata')),
        )
        return True

    def post_configure(self):
        self.post_configure_root_layer()

    def post_configure_root_layer(self):
        self.rootLayer = None

        uid = self.cfg('rootLayerUid')
        if not uid:
            return

        self.rootLayer = self.root.get(uid, gws.ext.object.layer)
        if not self.rootLayer:
            raise gws.NotFoundError(f'root layer {uid!r} not found')

        prj = self.rootLayer.find_closest(gws.ext.object.project)
        if not self.project:
            self.project = prj
            return

        if self.project != prj:
            raise gws.NotFoundError(f'root layer {uid!r} does not belong to {self.project!r}')

    ##

    def url_path(self) -> str:
        return gws.u.action_url_path('owsService', serviceUid=self.uid)

    ##

    def init_request(self, req: gws.WebRequester) -> request.Object:
        return request.Object(self, req)

    def handle_request(self, req: gws.WebRequester) -> gws.ContentResponse:
        sr = self.init_request(req)
        return self.dispatch_request(sr)

    def dispatch_request(self, sr: request.Object):
        handler = self.handlers.get(sr.verb)
        if not handler:
            raise error.OperationNotSupported()
        return handler(sr)

    def template_response(self, sr: request.Object, **kwargs) -> gws.ContentResponse:
        mime = None
        fmt = kwargs.pop('format', '')

        if fmt:
            mime = gws.lib.mime.get(fmt)
            if not mime:
                gws.log.debug(f'no mimetype: {sr.verb=} {fmt=}')
                raise error.InvalidFormat()

        tpl = self.root.app.templateMgr.find_template(self, user=sr.req.user, subject=f'ows.{sr.verb}', mime=mime)
        if not tpl:
            gws.log.debug(f'no template: {sr.verb=} {fmt=}')
            raise error.InvalidFormat()

        args = request.TemplateArgs(
            sr=sr,
            service=self,
            serviceUrl=self.url_path(),
            url_for=sr.req.url_for,
            version=sr.version,
            intVersion=int(sr.version.replace('.', '')),
            **kwargs,
        )

        return cast(gws.ContentResponse, tpl.render(gws.TemplateRenderInput(args=args)))
