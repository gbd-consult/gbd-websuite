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


class _Payload(gws.Data):
    step: int
    num_steps: int
    step_type: str
    step_name: str
    params_path: str
    result_path: str
    project_uid: str


def start_job(root: gws.IRoot, req: gws.IWebRequester, params: core.Params) -> gws.lib.job.Job:
    job = _create(root, req, params)
    gws.server.spool.add(job)
    return job


def run_job(root: gws.IRoot, params: core.Params, project_uid: str, user: gws.IUser):
    w = _Worker(root, '', project_uid, params, user)
    return w.run()


def job_status(job: gws.lib.job.Job) -> core.StatusResponse:
    payload = t.cast(_Payload, job.payload)

    def _progress():
        if job.state == gws.lib.job.State.complete:
            return 100
        if job.state != gws.lib.job.State.running:
            return 0
        num_steps = payload.num_steps or 0
        if not num_steps:
            return 0
        step = payload.step or 0
        return min(100.0, step * 100.0 / num_steps)

    return core.StatusResponse(
        jobUid=job.uid,
        state=job.state,
        progress=_progress(),
        steptype=payload.step_type or '',
        stepname=payload.step_name or '',
        url=gws.action_url_path('printerGetResult', jobUid=job.uid, projectUid=payload.project_uid)
    )


def job_result_path(job: gws.lib.job.Job) -> str:
    payload = t.cast(_Payload, job.payload)
    return payload.result_path or ''


##


def _create(root: gws.IRoot, req: gws.IWebRequester, params: core.Params) -> gws.lib.job.Job:
    job_uid = gws.random_string(64)
    params_path = gws.tempname('print.pickle')
    gws.serialize_to_path(params, params_path)

    return gws.lib.job.create(
        root,
        uid=job_uid,
        user=req.user,
        payload=_Payload(project_uid=params.projectUid, params_path=params_path),
        worker=__name__ + '._worker')


def _worker(root: gws.IRoot, job: gws.lib.job.Job):
    payload = t.cast(_Payload, job.payload)
    params = gws.unserialize_from_path(payload.params_path)

    job.update(state=gws.lib.job.State.running)

    w = _Worker(root, job.uid, payload.project_uid, params, job.user)
    try:
        w.run()
    except gws.lib.job.PrematureTermination as e:
        gws.log.warn(f'job={job.uid} TERMINATED {e.args!r}')


_PAPER_COLOR = 'white'


class _Worker:
    def __init__(self, root: gws.IRoot, job_uid: str, project_uid: str, params: core.Params, user: gws.IUser):
        self.job_uid = job_uid
        self.project: gws.IProject = user.require('gws.base.project', project_uid)
        self.root = root
        self.user = user

        self.page_count = 0

        self.tri = gws.TemplateRenderInput()
        self.tri.user = self.user

        fmt = params.outputFormat or 'pdf'
        if fmt.lower() == 'pdf' or fmt == gws.lib.mime.PDF:
            self.tri.out_mime = gws.lib.mime.PDF
        elif fmt.lower() == 'png' or fmt == gws.lib.mime.PNG:
            self.tri.out_mime = gws.lib.mime.PNG
        else:
            raise gws.Error(f'invalid outputFormat {fmt!r}')

        s = params.localeUid or ''
        self.tri.locale_uid = s if s in self.project.locale_uids else self.project.locale_uids[0]

        self.tri.crs = gws.gis.crs.get(params.crs) or self.project.map.crs
        self.tri.maps = [self.prepare_map(m) for m in (params.maps or [])]

        if params.type == 'template':
            self.template: gws.ITemplate = self.user.require('gws.ext.template', params.templateUid)
            try:
                ql = self.template.quality_levels[params.qualityLevel or 0]
            except IndexError:
                ql = self.template.quality_levels[0]
            self.tri.dpi = ql.dpi
        else:
            self.tri.dpi = min(gws.gis.render.MAX_DPI, max(params.dpi, gws.lib.uom.OGC_SCREEN_PPI))
            mm = gws.lib.uom.size_px_to_mm(params.outputSize, gws.lib.uom.OGC_SCREEN_PPI)
            px = gws.lib.uom.size_mm_to_px(mm, self.tri.dpi)
            w, h = px
            self.template: gws.ITemplate = self.create_child(
                'gws.ext.template.map',
                gws.Config(pageSize=(w, h, units.PX)))

        ctx = params.context or {}
        if self.template.data_model:
            atts = self.template.data_model.apply_to_dict(ctx)
            ctx = {a.name: a.value for a in atts}

        extra = {
            'project': self.project,
            'user': self.user,
            'localeUid': self.tri.locale_uid,
        }

        if self.tri.maps:
            extra['scale'] = self.tri.maps[0].scale
            extra['rotation'] = self.tri.maps[0].rotation

        self.tri.context = gws.merge(ctx, extra)

        num_steps = sum(len(mp.planes) for mp in self.tri.maps) + 1

        job = self.get_job()
        if job:
            payload = t.cast(_Payload, job.payload)
            payload.num_steps = num_steps
            job.update(payload=payload)

    def notify(self, event, details=None):
        gws.log.debug(f'print.worker.notify {event!r} {details!r}')

        job = self.get_job()
        if not job:
            return
        payload = t.cast(_Payload, job.payload)

        if event == 'begin_plane':
            name = gws.get(details, 'layer.title')
            payload.step = (payload.step or 0) + 1
            payload.step_type = 'begin_plane'
            payload.step_name = name or ''
            job.update(payload=payload)
            return

        if event == 'finalize_print':
            payload.step = (payload.step or 0) + 1
            job.update(payload=payload)
            return

        if event == 'begin_page':
            self.page_count += 1
            payload.step = 0
            payload.step_type = 'begin_page'
            payload.step_name = str(self.page_count)
            job.update(payload=payload)
            return

    def run(self):
        resp = self.template.render(self.tri, self.notify)

        job = self.get_job()
        if job:
            payload = t.cast(_Payload, job.payload)
            payload.result_path = resp.path
            job.update(state=gws.lib.job.State.complete, payload=payload)

        return resp.path

    def prepare_map(self, mp: core.MapParams) -> gws.TemplateRenderInputMap:
        planes = []

        if mp.planes:
            for n, p in enumerate(mp.planes):
                pp = self.prepare_plane(p)
                if not pp:
                    gws.log.warn(f'render plane {n} FAILED')
                    continue
                planes.append(pp)

        return gws.TemplateRenderInputMap(
            background_color=_PAPER_COLOR,
            bbox=mp.bbox,
            center=mp.center,
            planes=planes,
            rotation=mp.rotation or 0,
            scale=mp.scale,
            visible_layers=gws.compact(self.user.acquire('gws.ext.layer', uid) for uid in (mp.visibleLayers or []))
        )

    def prepare_plane(self, plane: core.Plane) -> t.Optional[gws.MapRenderInputPlane]:
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

        if plane.type == 'raster':
            layer = self.user.acquire('gws.ext.layer', plane.layerUid)
            if not layer or not layer.can_render_box:
                return
            return gws.MapRenderInputPlane(
                type='image_layer',
                layer=layer,
                opacity=opacity,
                style=style,
                sub_layers=plane.get('subLayers'))

        if plane.type == 'vector':
            layer = self.user.acquire('gws.ext.layer', plane.layerUid)
            if not layer or not layer.can_render_svg:
                return
            return gws.MapRenderInputPlane(
                type='svg_layer',
                layer=layer,
                opacity=opacity,
                style=style)

        if plane.type == 'bitmap':
            img = None
            if plane.mode in ('RGBA', 'RGB'):
                img = gws.lib.image.from_raw_data(plane.data, plane.mode, (plane.width, plane.height))
            if not img:
                return
            return gws.MapRenderInputPlane(
                type='image',
                image=img,
                opacity=opacity,
                style=style)

        if plane.type == 'url':
            img = gws.lib.image.from_data_url(plane.url)
            if not img:
                return
            return gws.MapRenderInputPlane(
                type='image',
                image=img,
                opacity=opacity,
                style=style)

        if plane.type == 'features':
            fs = []
            for p in plane.features:
                f = gws.base.feature.from_props(p)
                if f and f.shape:
                    fs.append(f)
            if not fs:
                return
            return gws.MapRenderInputPlane(
                type='features',
                features=fs,
                opacity=opacity,
                style=style)

        if plane.type == 'soup':
            return gws.MapRenderInputPlane(
                type='svg_soup',
                soup_points=plane.points,
                soup_tags=plane.tags,
                opacity=opacity,
                style=style)

        raise gws.Error(f'invalid plane type {plane.type!r}')

    def get_job(self) -> t.Optional[gws.lib.job.Job]:
        if not self.job_uid:
            return None

        job = gws.lib.job.get_for(self.root, self.user, self.job_uid)

        if not job:
            raise gws.lib.job.PrematureTermination('JOB_NOT_FOUND')

        if job.state != gws.lib.job.State.running:
            raise gws.lib.job.PrematureTermination(f'JOB_WRONG_STATE={job.state!r}')

        return job
