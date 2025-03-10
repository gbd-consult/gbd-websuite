"""OWS Service."""

from typing import Optional, cast

import gws
import gws.base.legend
import gws.base.template
import gws.base.web
import gws.config.util
import gws.gis.render
import gws.gis.source
import gws.lib.bounds
import gws.lib.crs
import gws.lib.datetimex
import gws.lib.extent
import gws.lib.gml
import gws.lib.image
import gws.lib.metadata
import gws.lib.mime
import gws.lib.xmlx
from . import core, error, request


class Config(gws.ConfigWithAccess):
    defaultFeatureCount: int = 1000
    """Default number of features per page."""
    extent: Optional[gws.Extent]
    """Service extent."""
    imageFormats: Optional[list[gws.ImageFormatConfig]]
    """Supported image formats. (added in 8.1)"""
    maxFeatureCount: int = 10000
    """Max number of features per page. (added in 8.1)"""
    metadata: Optional[gws.Metadata]
    """Service metadata."""
    rootLayerUid: str = ''
    """Root layer uid."""
    searchLimit: int = 10000
    """Search limit. (deprecated in 8.1)"""
    searchTolerance: gws.UomValueStr = '10px'
    """Search pixel tolerance."""
    supportedCrs: Optional[list[gws.CrsName]]
    """List of CRS supported by this service."""
    templates: Optional[list[gws.ext.config.template]]
    """XML and HTML templates."""
    updateSequence: Optional[str]
    """Service update sequence."""
    withInspireMeta: bool = False
    """Emit INSPIRE Metadata."""
    withStrictParams: bool = False
    """Use strict params checking."""


class Object(gws.OwsService):
    """Baseclass for OWS services."""

    def configure(self):
        self.project = self.find_closest(gws.ext.object.project)

        self.updateSequence = self.cfg('updateSequence')
        self.withInspireMeta = self.cfg('withInspireMeta')
        self.withStrictParams = self.cfg('withStrictParams')

        self.maxFeatureCount = self.cfg('maxFeatureCount')
        self.defaultFeatureCount = self.cfg('defaultFeatureCount')
        self.searchTolerance = self.cfg('searchTolerance')

        self.configure_bounds()
        self.configure_image_formats()
        self.configure_templates()
        self.configure_operations()
        self.configure_metadata()

    def configure_image_formats(self):
        p = self.cfg('imageFormats')
        if p:
            self.imageFormats = []
            for cfg in p:
                self.imageFormats.append(gws.ImageFormat(
                    mimeTypes=[s.replace(' ', '') for s in cfg.get('mimeTypes', [])],
                    options=cfg.get('options') or {}
                ))
            return

        self.imageFormats = [
            gws.ImageFormat(mimeTypes=[gws.lib.mime.PNG], options={}),
            gws.ImageFormat(mimeTypes=[gws.lib.mime.JPEG], options={}),
        ]

    def configure_bounds(self):
        p = self.cfg('supportedCrs')
        if p:
            crs_list = [gws.lib.crs.require(s) for s in p]
        elif self.project:
            crs_list = [self.project.map.bounds.crs]
        else:
            crs_list = [gws.lib.crs.WEBMERCATOR, gws.lib.crs.WGS84]

        p = self.cfg('extent')
        if p:
            bounds = gws.Bounds(crs=crs_list[0], extent=gws.lib.extent.from_list(p))
        elif self.project:
            bounds = self.project.map.bounds
        else:
            bounds = gws.Bounds(crs=crs_list[0], extent=crs_list[0].extent)

        self.supportedBounds = [gws.lib.bounds.transform(bounds, crs) for crs in crs_list]
        return True

    def configure_templates(self):
        return gws.config.util.configure_templates_for(self)

    def configure_operations(self):
        pass

    def available_formats(self, verb: gws.OwsVerb):
        fs = set()

        if verb in core.IMAGE_VERBS:
            for fmt in self.imageFormats:
                fs.update(fmt.mimeTypes)
        else:
            for tpl in self.templates:
                if tpl.subject == f'ows.{verb}':
                    fs.update(tpl.mimeTypes)

        return sorted(fs)

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

    def url_path(self, sr: request.Object) -> str:
        if sr.project:
            return gws.u.action_url_path('owsService', serviceUid=self.uid, projectUid=sr.project.uid)
        else:
            return gws.u.action_url_path('owsService', serviceUid=self.uid)

    ##

    def init_request(self, req: gws.WebRequester) -> request.Object:
        if req.method == 'GET':
            return request.Object(self, req, req.params())

        if req.method == 'POST' and gws.lib.mime.get(req.contentType) == gws.lib.mime.XML:
            try:
                xml = gws.lib.xmlx.from_string(req.text(), remove_namespaces=True, case_insensitive=True)
            except gws.lib.xmlx.Error:
                raise gws.base.web.error.BadRequest()

            is_soap = False
            if xml.lcName == 'envelope':
                is_soap = True
                try:
                    xml = xml.findfirst('body').findfirst()
                except gws.lib.xmlx.Error:
                    raise gws.base.web.error.BadRequest()

            params = self.parse_xml_request(xml)
            if not params:
                raise gws.base.web.error.BadRequest()
            return request.Object(self, req, params, xml_element=xml, is_soap=is_soap)

        # @TODO support application/x-www-form-urlencoded
        raise gws.base.web.error.BadRequest()

    def parse_xml_request(self, xml: gws.XmlElement) -> Optional[dict]:
        return {}

    def handle_request(self, req: gws.WebRequester) -> gws.ContentResponse:
        try:
            sr = self.init_request(req)
            return self.dispatch_request(sr)
        except Exception as exc:
            err = error.from_exception(exc)
            # @TODO INIMAGE Exceptions
            # verb = req.param('REQUEST')
            # if verb in core.IMAGE_VERBS:
            #     return err.to_image_response()
            return err.to_xml_response('ows' if self.isOwsCommon else 'ogc')

    def dispatch_request(self, sr: request.Object):
        fn = getattr(self, sr.operation.handlerName)
        return fn(sr)

    def template_response(self, sr: request.Object, mime: str = '', **kwargs) -> gws.ContentResponse:
        tpl = self.root.app.templateMgr.find_template(
            f'ows.{sr.operation.verb}',
            where=[self, sr.project],
            user=sr.req.user,
            mime=mime,
        )
        if not tpl:
            # OGC 06-042, 7.2.3.1
            # If the request specifies a format not supported by the server, the server shall respond with the default text/xml format.
            gws.log.debug(f'no template: {sr.operation.verb=} {mime=}')
            if mime == gws.lib.mime.XML:
                raise error.InvalidFormat()
            return self.template_response(sr, gws.lib.mime.XML, **kwargs)

        args = request.TemplateArgs(
            sr=sr,
            service=self,
            serviceUrl=sr.req.url_for(self.url_path(sr)),
            url_for=sr.req.url_for,
            version=sr.version,
            intVersion=int(sr.version.replace('.', '')),
            **kwargs,
        )

        return tpl.render(gws.TemplateRenderInput(args=args))

    def image_response(self, sr: request.Object, img: Optional[gws.Image], mime: str) -> gws.ContentResponse:
        ifmt = self.find_image_format(mime)
        if img:
            gws.log.debug(f'image_response: {img.mode()=} {img.size()=} {mime=} {ifmt.options}')
        content = img.to_bytes(mime, ifmt.options) if img else gws.lib.image.empty_pixel(mime)
        return gws.ContentResponse(mime=mime, content=content)

    def find_image_format(self, mime: str) -> gws.ImageFormat:
        if not mime:
            return self.imageFormats[0]
        for f in self.imageFormats:
            if mime in f.mimeTypes:
                return f
        raise error.InvalidFormat()

    def render_legend(self, sr: request.Object, lcs: list[core.LayerCaps], mime: str) -> gws.ContentResponse:
        uids = [lc.layer.uid for lc in lcs]
        cache_key = 'gws.base.ows.server.legend.' + gws.u.sha256(uids) + mime

        def _get():
            legend = cast(gws.Legend, self.root.create_temporary(
                gws.ext.object.legend,
                type='combined',
                layerUids=uids,
            ))
            lro = legend.render()
            if not lro:
                return self.image_response(sr, None, mime)
            return self.image_response(sr, gws.base.legend.output_to_image(lro), mime)

        return gws.u.get_app_global(cache_key, _get)

    def feature_collection(self, sr: request.Object, lcs: list[core.LayerCaps], hits: int, results: list[gws.SearchResult]) -> core.FeatureCollection:
        fc = core.FeatureCollection(
            members=[],
            timestamp=gws.lib.datetimex.now(),
            numMatched=hits,
            numReturned=len(results),
        )

        lcs_map = {id(lc.layer): lc for lc in lcs}

        for r in results:
            r.feature.transform_to(sr.targetCrs)
            fc.members.append(core.FeatureCollectionMember(
                feature=r.feature,
                layer=r.layer,
                layerCaps=lcs_map.get(id(r.layer)) if r.layer else None
            ))

        return fc
