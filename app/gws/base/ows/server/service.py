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

from . import core, util


class Config(gws.ConfigWithAccess):
    extent: Optional[gws.Extent]
    """service extent"""
    metadata: Optional[gws.Metadata]
    """service metadata"""
    rootLayerUid: str = ''
    """root layer uid"""
    searchLimit: int = 100
    """max search limit"""
    searchTolerance: gws.UomValueStr = '10px'
    """search pixel tolerance"""
    supportedCrs: Optional[list[gws.CrsName]]
    """supported CRS for this service"""
    templates: Optional[list[gws.ext.config.template]]
    """service XML templates"""
    updateSequence: Optional[str]
    """service update sequence"""
    withInspireMeta: bool = False
    """use INSPIRE Metadata"""
    withStrictParams: bool = False
    """use strict params checking"""


class Object(gws.OwsService):
    """Baseclass for OWS services."""

    project: Optional[gws.Project]
    rootLayer: Optional[gws.Layer]

    isRasterService = False
    isVectorService = False

    searchMaxLimit: int
    searchTolerance: gws.UomValue

    handlers: dict[gws.OwsVerb, Callable]

    OWS_VERBS = [
        gws.OwsVerb.CreateStoredQuery,
        gws.OwsVerb.DescribeCoverage,
        gws.OwsVerb.DescribeFeatureType,
        gws.OwsVerb.DescribeLayer,
        gws.OwsVerb.DescribeRecord,
        gws.OwsVerb.DescribeStoredQueries,
        gws.OwsVerb.DropStoredQuery,
        gws.OwsVerb.GetCapabilities,
        gws.OwsVerb.GetFeature,
        gws.OwsVerb.GetFeatureInfo,
        gws.OwsVerb.GetFeatureWithLock,
        gws.OwsVerb.GetLegendGraphic,
        gws.OwsVerb.GetMap,
        gws.OwsVerb.GetPrint,
        gws.OwsVerb.GetPropertyValue,
        gws.OwsVerb.GetRecordById,
        gws.OwsVerb.GetRecords,
        gws.OwsVerb.GetTile,
        gws.OwsVerb.ListStoredQueries,
        gws.OwsVerb.LockFeature,
        gws.OwsVerb.Transaction,
    ]

    def configure(self):
        self.project = self.find_closest(gws.ext.object.project)

        self.updateSequence = self.cfg('updateSequence')
        self.withInspireMeta = self.cfg('withInspireMeta')
        self.withStrictParams = self.cfg('withStrictParams')

        self.searchMaxLimit = self.cfg('searchLimit')
        self.searchTolerance = self.cfg('searchTolerance')

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
        fs = {}

        for tpl in self.templates:
            for mime in tpl.mimeTypes:
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

        prj = self.rootLayer.find_closest(gws.ext.object.project)
        if not self.project:
            self.project = prj
            return

        if self.project != prj:
            raise gws.Error(f'root layer {uid!r} does not belong to {self.project!r}')

    # Requests

    def handle_request(self, req: gws.WebRequester) -> gws.ContentResponse:
        sr = self.init_service_request(req)
        return self.dispatch_service_request(sr)

    ##

    def init_service_request(self, req: gws.WebRequester) -> core.ServiceRequest:
        sr = core.ServiceRequest(req=req, service=self)
        sr.alwaysXY = False
        sr.project = self.requested_project(sr)
        sr.isSoap = False
        sr.verb = self.requested_verb(sr)
        sr.version = self.requested_version(sr)

        # OGC 06-042, 7.2.3.5
        s = sr.req.param('updatesequence')
        if s and self.updateSequence and s >= self.updateSequence:
            raise gws.base.web.error.BadRequest('Wrong update sequence')

        return sr

    def requested_project(self, sr: core.ServiceRequest) -> Optional[gws.Project]:
        # services can be configured globally (in which case, self.project == None)
        # and applied to multiple projects with the projectUid param
        # or, configured just for a single project (self.project != None)

        p = sr.req.param('projectUid')
        if p:
            project = sr.req.user.require_project(p)
            if self.project and project != self.project:
                gws.log.debug(f'ows {self.uid=}: wrong project={p!r}')
                raise gws.base.web.error.NotFound('Project not found')
            return project

        if self.project:
            # for in-project services, ensure the user can access the project
            return sr.req.user.require_project(self.project.uid)

    def requested_version(self, sr: core.ServiceRequest) -> str:
        s = util.one_of_params(sr, 'version', 'acceptversions')
        if not s:
            # the first supported version is the default
            return self.supportedVersions[0]

        for v in gws.u.to_list(s):
            for ver in self.supportedVersions:
                if ver.startswith(v):
                    return ver

        raise gws.base.web.error.BadRequest('Unsupported service version')

    def requested_verb(self, sr: core.ServiceRequest) -> gws.OwsVerb:
        s = util.one_of_params(sr, 'request') or ''

        for verb in self.OWS_VERBS:
            if verb.lower() == s.lower():
                return verb

        raise gws.base.web.error.BadRequest('Invalid REQUEST parameter')

    def requested_crs(self, sr: core.ServiceRequest, *param_names) -> Optional[gws.Crs]:
        s = util.one_of_params(sr, *param_names)
        if s:
            crs = gws.gis.crs.get(s)
            if not crs:
                raise gws.base.web.error.BadRequest('Invalid CRS')
            if all(crs != b.crs for b in self.supportedBounds):
                raise gws.base.web.error.BadRequest('Unsupported CRS')
            return crs

    def requested_bounds(self, sr: core.ServiceRequest, *param_names) -> gws.Bounds:
        # OGC 06-042, 7.2.3.5
        # OGC 00-028, 6.2.8.2.3

        s = util.one_of_params(sr, *param_names)
        if s:
            bounds = gws.gis.bounds.from_request_bbox(s, default_crs=sr.crs, always_xy=sr.alwaysXY)
            if not bounds:
                raise gws.base.web.error.BadRequest('Invalid BBOX')
            return gws.gis.bounds.transform(bounds, sr.crs)

    ##

    def dispatch_service_request(self, sr: core.ServiceRequest):
        handler = self.handlers.get(sr.verb)
        if not handler:
            raise gws.base.web.error.BadRequest('Invalid REQUEST parameter')
        return handler(sr)

    def template_response(self, sr: core.ServiceRequest, **kwargs) -> gws.ContentResponse:
        mime = None
        format = kwargs.pop('format', '')

        if format:
            mime = gws.lib.mime.get(format)
            if not mime:
                gws.log.debug(f'no mimetype: {sr.verb=} {format=}')
                raise gws.base.web.error.BadRequest('Invalid FORMAT')

        tpl = self.root.app.templateMgr.find_template(self, user=sr.req.user, subject=f'ows.{sr.verb}', mime=mime)
        if not tpl:
            gws.log.debug(f'no template: {sr.verb=} {format=}')
            raise gws.base.web.error.BadRequest('Unsupported FORMAT')

        args = core.TemplateArgs(
            sr=sr,
            service=self,
            serviceUrl=sr.req.url_for(util.service_url_path(self, sr.project)),
            url_for=sr.req.url_for,
            version=sr.version,
            **kwargs,
        )

        return cast(gws.ContentResponse, tpl.render(gws.TemplateRenderInput(args=args)))
