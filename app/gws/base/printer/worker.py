from typing import Optional, cast

import gws
import gws.base.model
import gws.gis.crs
import gws.gis.render
import gws.lib.image
import gws.lib.intl
import gws.lib.job
import gws.lib.mime
import gws.lib.osx
import gws.lib.style
import gws.lib.uom


def worker(root: gws.Root, job: gws.lib.job.Object):
    request = gws.u.unserialize_from_path(job.payload.get('requestPath'))
    w = Object(root, job.uid, request, job.user)
    w.run()


_PAPER_COLOR = 'white'


class Object:
    jobUid: str
    user: gws.User
    project: gws.Project
    tri: gws.TemplateRenderInput
    printer: gws.Printer
    template: gws.Template

    def __init__(self, root: gws.Root, job_uid: str, request: gws.PrintRequest, user: gws.User):
        self.jobUid = job_uid
        self.root = root
        self.user = user

        self.project = cast(gws.Project, self.user.require(request.projectUid, gws.ext.object.project))

        self.page_count = 0

        self.tri = gws.TemplateRenderInput(
            project=self.project,
            user=self.user,
            notify=self.notify,
        )

        fmt = request.outputFormat or 'pdf'
        if fmt.lower() == 'pdf' or fmt == gws.lib.mime.PDF:
            self.tri.mimeOut = gws.lib.mime.PDF
        elif fmt.lower() == 'png' or fmt == gws.lib.mime.PNG:
            self.tri.mimeOut = gws.lib.mime.PNG
        else:
            raise gws.Error(f'invalid outputFormat {fmt!r}')

        self.tri.locale = gws.lib.intl.locale(request.localeUid, self.tri.project.localeUids)
        self.tri.crs = gws.gis.crs.get(request.crs) or self.project.map.bounds.crs
        self.tri.maps = [self.prepare_map(self.tri, m) for m in (request.maps or [])]
        self.tri.dpi = int(min(gws.gis.render.MAX_DPI, max(request.dpi, gws.lib.uom.OGC_SCREEN_PPI)))

        if request.type == 'template':
            # @TODO check dpi against configured qualityLevels
            self.printer = cast(gws.Printer, self.user.require(request.printerUid, gws.ext.object.printer))
            self.template = self.printer.template
        else:
            mm = gws.lib.uom.size_px_to_mm(request.outputSize, gws.lib.uom.OGC_SCREEN_PPI)
            px = gws.lib.uom.size_mm_to_px(mm, self.tri.dpi)
            self.template = self.root.create_temporary(
                gws.ext.object.template,
                type='map',
                pageSize=(px[0], px[1], gws.Uom.px))

        extra = dict(
            project=self.project,
            user=self.user,
            scale=self.tri.maps[0].scale if self.tri.maps else 0,
            rotation=self.tri.maps[0].rotation if self.tri.maps else 0,
            dpi=self.tri.dpi,
            crs=self.tri.crs.srid,
            templateRenderInput=self.tri,
        )

        # @TODO read the args feature from the request
        self.tri.args = gws.u.merge(request.args, extra)

        num_steps = sum(len(mp.planes) for mp in self.tri.maps) + 1

        self.update_job(state=gws.JobState.running, numSteps=num_steps)

    def notify(self, event, details=None):
        gws.log.debug(f'JOB {self.jobUid}: print.worker.notify {event=} {details=}')
        args = {
            'stepType': event,
        }

        if event == 'begin_plane':
            name = gws.u.get(details, 'layer.title')
            args['progress'] = True
            args['stepName'] = name or ''

        elif event == 'finalize_print':
            args['progress'] = True
            return self.update_job(inc=True)

        elif event == 'begin_page':
            self.page_count += 1
            args['step'] = 0
            args['stepName'] = str(self.page_count)

        return self.update_job(**args)

    def run(self):
        res = self.template.render(self.tri)
        self.update_job(state=gws.JobState.complete, resultPath=res.contentPath)
        return res.contentPath

    def prepare_map(self, tri: gws.TemplateRenderInput, mp: gws.PrintMap) -> gws.MapRenderInput:
        planes = []

        style_opts = gws.lib.style.parser.Options(
            trusted=False,
            strict=False,
            imageDirs=[self.root.app.webMgr.sites[0].staticRoot.dir],
        )
        style_dct = {}

        for p in (mp.styles or []):
            style = gws.lib.style.from_props(p, style_opts)
            if style:
                style_dct[style.cssSelector] = style

        if mp.planes:
            for n, p in enumerate(mp.planes):
                pp = self.prepare_map_plane(n, p, style_dct)
                if pp:
                    planes.append(pp)

        layers = gws.u.compact(
            self.user.acquire(uid, gws.ext.object.layer)
            for uid in (mp.visibleLayers or []))

        return gws.MapRenderInput(
            backgroundColor=_PAPER_COLOR,
            bbox=mp.bbox,
            center=mp.center,
            crs=tri.crs,
            dpi=tri.dpi,
            notify=tri.notify,
            planes=planes,
            project=tri.project,
            rotation=mp.rotation or 0,
            scale=mp.scale,
            user=tri.user,
            visibleLayers=layers,
        )

    def prepare_map_plane(self, n, plane: gws.PrintPlane, style_dct) -> Optional[gws.MapRenderInputPlane]:
        opacity = 1
        s = plane.get('opacity')
        if s is not None:
            opacity = float(s)
            if opacity == 0:
                return

        if plane.type == gws.PrintPlaneType.raster:
            layer = cast(gws.Layer, self.user.acquire(plane.layerUid, gws.ext.object.layer))
            if not layer:
                gws.log.warning(f'PREPARE_FAILED: plane {n}: {plane.layerUid=} not found')
                return
            if not layer.canRenderBox:
                gws.log.warning(f'PREPARE_FAILED: plane {n}: {plane.layerUid=} canRenderBox false')
                return
            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.imageLayer,
                layer=layer,
                opacity=opacity,
                subLayers=plane.get('subLayers'),
            )

        if plane.type == gws.PrintPlaneType.vector:
            layer = cast(gws.Layer, self.user.acquire(plane.layerUid, gws.ext.object.layer))
            if not layer:
                gws.log.warning(f'PREPARE_FAILED: plane {n}: {plane.layerUid=} not found')
                return
            if not layer.canRenderSvg:
                gws.log.warning(f'PREPARE_FAILED: plane {n}: {plane.layerUid=} canRenderSvg false')
                return
            style = style_dct.get(plane.cssSelector)
            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.svgLayer,
                layer=layer,
                opacity=opacity,
                styles=[style] if style else [],
            )

        if plane.type == gws.PrintPlaneType.bitmap:
            img = None
            if plane.bitmapMode in ('RGBA', 'RGB'):
                img = gws.lib.image.from_raw_data(
                    plane.bitmapData,
                    plane.bitmapMode,
                    (plane.bitmapWidth, plane.bitmapHeight))
            if not img:
                gws.log.warning(f'PREPARE_FAILED: plane {n}: bitmap error')
                return
            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.image,
                image=img,
                opacity=opacity,
            )

        if plane.type == gws.PrintPlaneType.url:
            img = gws.lib.image.from_data_url(plane.url)
            if not img:
                gws.log.warning(f'PREPARE_FAILED: plane {n}: url error')
                return
            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.image,
                image=img,
                opacity=opacity,
            )

        if plane.type == gws.PrintPlaneType.features:
            model = self.root.app.modelMgr.default_model()
            used_styles = {}

            mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.map, user=self.user)
            features = []

            for props in plane.features:
                feature = model.feature_from_props(props, mc)
                if not feature or not feature.shape():
                    continue
                feature.cssSelector = feature.cssSelector or plane.cssSelector
                if feature.cssSelector in style_dct:
                    used_styles[feature.cssSelector] = style_dct[feature.cssSelector]
                features.append(feature)

            if not features:
                gws.log.warning(f'PREPARE_FAILED: plane {n}: no features')
                return

            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.features,
                features=features,
                opacity=opacity,
                styles=list(used_styles.values()),
            )

        if plane.type == gws.PrintPlaneType.soup:
            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.svgSoup,
                soupPoints=plane.soupPoints,
                soupTags=plane.soupTags,
                opacity=opacity,
            )

        raise gws.Error(f'invalid plane type {plane.type!r}')

    def update_job(self, **kwargs):
        job = self.get_job()
        if not job:
            return

        state = kwargs.pop('state', None)

        if kwargs.pop('progress', None):
            kwargs['step'] = job.payload.get('step', 0) + 1

        job.update(state=state, payload=gws.u.merge(job.payload, kwargs))

    def get_job(self) -> Optional[gws.lib.job.Object]:
        if not self.jobUid:
            return None

        job = gws.lib.job.get(self.root, self.jobUid)

        if not job:
            raise gws.lib.job.PrematureTermination('JOB_NOT_FOUND')
        if job.user.uid != self.user.uid:
            raise gws.lib.job.PrematureTermination('WRONG_USER {job.user.uid}')
        if job.state != gws.JobState.running:
            raise gws.lib.job.PrematureTermination(f'JOB_WRONG_STATE={job.state!r}')

        return job
