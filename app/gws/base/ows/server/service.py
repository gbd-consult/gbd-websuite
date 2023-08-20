"""OWS Service."""

import gws
import gws.base.layer.core
import gws.base.template
import gws.base.web
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

import gws.types as t

from . import core, util


class Config(gws.ConfigWithAccess):
    extent: t.Optional[gws.Extent]
    """service extent"""
    metadata: t.Optional[gws.Metadata]
    """service metadata"""
    rootLayerUid: str = ''
    """root layer uid"""
    searchLimit: int = 100
    """max search limit"""
    searchTolerance: int = 10
    """search pixel tolerance"""
    supportedCrs: t.Optional[list[gws.CrsName]]
    """supported CRS for this service"""
    templates: t.Optional[list[gws.ext.config.template]]
    """service XML templates"""
    updateSequence: t.Optional[str]
    """service update sequence"""
    withInspireMeta: bool = False
    """use INSPIRE Metadata"""
    withStrictParams: bool = False
    """use strict params checking"""


class Object(gws.Node, gws.IOwsService):
    """Baseclass for OWS services."""

    project: t.Optional[gws.IProject]
    rootLayer: t.Optional[gws.ILayer]

    isRasterService = False
    isVectorService = False

    searchMaxLimit: int
    searchTolerance: gws.Measurement

    # Configuration

    def configure(self):
        self.project = self.closest(gws.ext.object.project)

        self.updateSequence = self.cfg('updateSequence')
        self.withInspireMeta = self.cfg('withInspireMeta')
        self.withStrictParams = self.cfg('withStrictParams')

        self.searchMaxLimit = self.cfg('searchLimit')
        self.searchTolerance = self.cfg('searchTolerance'), gws.Uom.px

        self.configure_bounds()
        self.configure_templates()
        self.configure_operations()
        self.configure_metadata()

    def configure_bounds(self):
        crs_list = [gws.gis.crs.require(s) for s in self.cfg('supportedCrs', default=[])]
        if not crs_list:
            crs_list = [self.project.map.bounds.crs] if self.project else gws.gis.crs.WEBMERCATOR

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
        self.templates = gws.compact(self.configure_template(c) for c in self.cfg('templates', default=[]))
        return True

    def configure_template(self, cfg):
        return self.create_child(gws.ext.object.template, cfg)

    def configure_operations(self):
        fs = {}

        for tpl in self.templates:
            for mime in tpl.mimes:
                s = tpl.subject.split('.')
                fs.setdefault(s[-1], set()).add(mime)

        self.supportedOperations = [
            gws.OwsOperation(formats=sorted(formats), verb=verb)
            for verb, formats in fs.items()
        ]

        return True

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
            raise gws.Error(f'root layer {uid!r} not found')

        prj = self.rootLayer.closest(gws.ext.object.project)
        if not self.project:
            self.project = prj
            return

        if self.project != prj:
            raise gws.Error(f'root layer {uid!r} does not belong to {self.project!r}')

    # Requests

    def handle_request(self, req: gws.IWebRequester) -> gws.ContentResponse:
        rd = self.init_request(req)
        return self.dispatch_request(rd, req.param('request', ''))

    def init_request(self, req: gws.IWebRequester) -> core.Request:
        rd = core.Request(req=req, service=self)
        rd.project = self.requested_project(rd)
        rd.version = self.requested_version(rd)
        return rd

    def dispatch_request(self, rd: core.Request, verb: str):
        handler = getattr(self, 'handle_' + verb.lower(), None)
        if not handler:
            gws.log.debug(f'ows {self.uid=}: {verb=} not found')
            raise gws.base.web.error.BadRequest('Invalid REQUEST parameter')
        return handler(rd)

    def requested_project(self, rd: core.Request) -> t.Optional[gws.IProject]:
        # services can be configured globally (in which case, self.project == None)
        # and applied to multiple projects with the projectUid param
        # or, configured just for a single project (self.project != None)

        p = rd.req.param('projectUid')
        if p:
            project = rd.req.require_project(p)
            if self.project and project != self.project:
                gws.log.debug(f'ows {self.uid=}: wrong project={p!r}')
                raise gws.base.web.error.NotFound('Project not found')
            return project

        if self.project:
            # for in-project services, ensure the user can access the project
            return rd.req.require_project(self.project.uid)

    def requested_version(self, rd: core.Request) -> str:
        s = util.one_of_params(rd, 'version', 'acceptversions')
        if not s:
            # the first supported version is the default
            return self.supportedVersions[0]

        for v in gws.to_list(s):
            for ver in self.supportedVersions:
                if ver.startswith(v):
                    return ver

        raise gws.base.web.error.BadRequest('Unsupported service version')

    def requested_crs(self, rd: core.Request) -> t.Optional[gws.ICrs]:
        s = util.one_of_params(rd, 'crs', 'srs', 'crsName', 'srsName')
        if s:
            crs = gws.gis.crs.get(s)
            if not crs:
                raise gws.base.web.error.BadRequest('Invalid CRS')
            if all(crs != b.crs for b in self.supportedBounds):
                raise gws.base.web.error.BadRequest('Invalid CRS')
            return crs

    def requested_bounds(self, rd: core.Request) -> gws.Bounds:
        # OGC 06-042, 7.2.3.5
        # OGC 00-028, 6.2.8.2.3

        bounds = gws.gis.bounds.from_request_bbox(
            rd.req.param('bbox'),
            default_crs=rd.crs,
            always_xy=rd.alwaysXY)

        if not bounds:
            raise gws.base.web.error.BadRequest('Invalid BBOX')

        return gws.gis.bounds.transform(bounds, rd.crs)

    # Rendering and responses

    def template_response(self, rd: core.Request, verb: gws.OwsVerb, format_name: str = '', **kwargs) -> gws.ContentResponse:
        mime = None

        if format_name:
            mime = gws.lib.mime.get(format_name)
            if not mime:
                gws.log.debug(f'no mimetype: {verb=} {format_name=}')
                raise gws.base.web.error.BadRequest('Invalid FORMAT')

        tpl = gws.base.template.locate(self, user=rd.req.user, subject=f'ows.{verb}', mime=mime)
        if not tpl:
            gws.log.debug(f'no template: {verb=} {format_name=}')
            raise gws.base.web.error.BadRequest('Unsupported FORMAT')

        ta = gws.Data(
            project=rd.project,
            request=rd,
            service=self,
            serviceUrl=rd.req.url_for(util.service_url_path(self, rd.project)),
            url_for=rd.req.url_for,
            version=rd.version,
            **kwargs,
        )

        return tpl.render(gws.TemplateRenderInput(args={'owsArgs': ta}))
