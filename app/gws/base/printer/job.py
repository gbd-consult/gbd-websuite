import base64
import os

import gws
import gws.gis.crs
import gws.base.feature
import gws.lib.image
import gws.lib.job
import gws.lib.mime
import gws.lib.osx
import gws.gis.render
import gws.lib.style
import gws.lib.uom as units
import gws.server.spool

import gws.types as t

from . import core


def start(root: gws.IRoot, request: core.Request, user: gws.IUser) -> gws.lib.job.Job:
    request_path = gws.serialize_to_path(request, gws.printtemp('print.pickle'))

    job = gws.lib.job.create(
        root,
        user=user,
        payload=dict(requestPath=request_path, projectUid=request.projectUid),
        worker=__name__ + '._worker')

    if gws.server.spool.is_active():
        gws.server.spool.add(job)
    else:
        gws.lib.job.run(root, job.uid)
    return gws.lib.job.get(root, job.uid)


def run(root: gws.IRoot, request: core.Request, user: gws.IUser):
    w = _Worker(root, '', request, user)
    return w.run()


def status(job: gws.lib.job.Job) -> core.StatusResponse:
    payload = job.payload

    def _progress():
        if job.state == gws.lib.job.State.complete:
            return 100
        if job.state != gws.lib.job.State.running:
            return 0
        num_steps = payload.get('numSteps', 0)
        if not num_steps:
            return 0
        step = payload.get('step', 0)
        return int(min(100.0, step * 100.0 / num_steps))

    return core.StatusResponse(
        jobUid=job.uid,
        state=job.state,
        progress=_progress(),
        stepType=payload.get('stepType', ''),
        stepName=payload.get('stepName', ''),
        url=gws.action_url_path('printerResult', jobUid=job.uid, projectUid=payload.get('projectUid'))
    )


def result_path(job: gws.lib.job.Job) -> str:
    return job.payload.get('resultPath')


##


def _worker(root: gws.IRoot, job: gws.lib.job.Job):
    request = gws.unserialize_from_path(job.payload.get('requestPath'))
    w = _Worker(root, job.uid, request, job.user)
    w.run()


_PAPER_COLOR = 'white'


class _Worker:
    jobUid: str
    user: gws.IUser
    project: gws.IProject
    tri: gws.TemplateRenderInput
    template: gws.ITemplate

    def __init__(self, root: gws.IRoot, job_uid: str, request: core.Request, user: gws.IUser):
        self.jobUid = job_uid
        self.root = root
        self.user = user

        self.project = t.cast(gws.IProject, self.user.require(request.projectUid, gws.ext.object.project))

        self.page_count = 0

        self.tri = gws.TemplateRenderInput(
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

        s = request.localeUid or ''
        self.tri.localeUid = s if s in self.project.localeUids else self.project.localeUids[0]

        self.tri.crs = gws.gis.crs.get(request.crs) or self.project.map.bounds.crs
        self.tri.maps = [self.prepare_map(m) for m in (request.maps or [])]
        self.tri.dpi = int(min(gws.gis.render.MAX_DPI, max(request.dpi, gws.lib.uom.OGC_SCREEN_PPI)))

        if request.type == 'template':
            # @TODO check dpi against configured qualityLevels
            self.template = t.cast(gws.ITemplate, self.user.require(request.templateUid, gws.ext.object.template))
        else:
            mm = gws.lib.uom.size_px_to_mm(request.outputSize, gws.lib.uom.OGC_SCREEN_PPI)
            px = gws.lib.uom.size_mm_to_px(mm, self.tri.dpi)
            self.template = self.root.create_temporary(
                gws.ext.object.template,
                type='map',
                pageSize=(px[0], px[1], gws.Uom.px))

        # if self.template.data_model:
        #     atts = self.template.data_model.apply_to_dict(ctx)
        #     ctx = {a.name: a.value for a in atts}

        extra = dict(
            project=self.project,
            user=self.user,
            localeUid=self.tri.localeUid,
            scale=self.tri.maps[0].scale if self.tri.maps else 0,
            rotation=self.tri.maps[0].rotation if self.tri.maps else 0,
            dpi=self.tri.dpi,
            crs=self.tri.crs.srid,
            templateRenderInput=self.tri,
        )

        self.tri.args = gws.merge(request.args, extra)

        num_steps = sum(len(mp.planes) for mp in self.tri.maps) + 1

        self.update_job(state=gws.lib.job.State.running, numSteps=num_steps)

    def notify(self, event, details=None):
        gws.log.debug(f'JOB {self.jobUid}: print.worker.notify {event=} {details=}')

        if event == 'begin_plane':
            name = gws.get(details, 'layer.title')
            return self.update_job(inc=True, stepType='begin_plane', stepName=name or '')

        if event == 'finalize_print':
            return self.update_job(inc=True)

        if event == 'begin_page':
            self.page_count += 1
            return self.update_job(step=0, stepType='begin_page', stepName=str(self.page_count))

    def run(self):
        resp = self.template.render(self.tri)
        self.update_job(state=gws.lib.job.State.complete, resultPath=resp.path)
        return resp.path

    def prepare_map(self, mp: core.MapParams) -> gws.MapRenderInput:
        planes = []

        if mp.planes:
            for n, p in enumerate(mp.planes):
                pp = self.prepare_map_plane(p)
                if not pp:
                    gws.log.warning(f'PREPARE_FAILED: plane {n}')
                    continue
                planes.append(pp)

        return gws.MapRenderInput(
            backgroundColor=_PAPER_COLOR,
            bbox=mp.bbox,
            center=mp.center,
            planes=planes,
            rotation=mp.rotation or 0,
            scale=mp.scale,
            visibleLayers=gws.compact(
                self.user.acquire(uid, gws.ext.object.layer)
                for uid in (mp.visibleLayers or []))
        )

    def prepare_map_plane(self, plane: core.Plane) -> t.Optional[gws.MapRenderInputPlane]:
        opacity = 1
        s = plane.get('opacity')
        if s is not None:
            opacity = float(s)
            if opacity == 0:
                return

        style = None
        s = plane.get('style')
        if s:
            style = gws.lib.style.from_props(s)

        if plane.type == core.PlaneType.raster:
            layer = t.cast(gws.ILayer, self.user.acquire(plane.layerUid, gws.ext.object.layer))
            if not layer or not layer.canRenderBox:
                return
            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.image_layer,
                layer=layer,
                opacity=opacity,
                style=style,
                subLayers=plane.get('subLayers'))

        if plane.type == core.PlaneType.vector:
            layer = t.cast(gws.ILayer, self.user.acquire(plane.layerUid, gws.ext.object.layer))
            if not layer or not layer.canRenderSvg:
                return
            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.svg_layer,
                layer=layer,
                opacity=opacity,
                style=style)

        if plane.type == core.PlaneType.bitmap:
            img = None
            if plane.bitmapMode in ('RGBA', 'RGB'):
                img = gws.lib.image.from_raw_data(
                    plane.bitmapData,
                    t.cast(gws.lib.image.ImageMode, plane.bitmapMode),
                    (plane.bitmapWidth, plane.bitmapHeight))
            if not img:
                return
            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.image,
                image=img,
                opacity=opacity,
                style=style)

        if plane.type == core.PlaneType.url:
            img = gws.lib.image.from_data_url(plane.url)
            if not img:
                return
            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.image,
                image=img,
                opacity=opacity,
                style=style)

        # if plane.type == core.PlaneType.features:
        #     fs = []
        #     for p in plane.features:
        #         f = gws.base.feature.from_props(p)
        #         if f and f.shape:
        #             fs.append(f)
        #     if not fs:
        #         return
        #     return gws.MapRenderInputPlane(
        #         type=gws.MapRenderInputPlaneType.features,
        #         features=fs,
        #         opacity=opacity,
        #         style=style)

        if plane.type == core.PlaneType.soup:
            return gws.MapRenderInputPlane(
                type=gws.MapRenderInputPlaneType.svg_soup,
                soup_points=plane.points,
                soup_tags=plane.tags,
                opacity=opacity,
                style=style)

        raise gws.Error(f'invalid plane type {plane.type!r}')

    def update_job(self, **kwargs):
        job = self.get_job()
        if not job:
            return

        state = kwargs.pop('state', None)

        if kwargs.pop('inc', None):
            kwargs['step'] = job.payload.get('step', 0) + 1

        job.update(state=state, payload=gws.merge(job.payload, kwargs))

    def get_job(self) -> t.Optional[gws.lib.job.Job]:
        if not self.jobUid:
            return None

        job = gws.lib.job.get_for(self.root, self.user, self.jobUid)

        if not job:
            raise gws.lib.job.PrematureTermination('JOB_NOT_FOUND')

        if job.state != gws.lib.job.State.running:
            raise gws.lib.job.PrematureTermination(f'JOB_WRONG_STATE={job.state!r}')

        return job
