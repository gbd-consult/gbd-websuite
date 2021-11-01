import base64
import os

import gws
import gws.base.template
import gws.lib.crs
import gws.lib.feature
import gws.lib.image
import gws.lib.job
import gws.lib.os2
import gws.lib.pdf
import gws.lib.render
import gws.lib.style
import gws.lib.units as units
import gws.server.spool
import gws.types as t

from . import types


class PreparedSection(gws.Data):
    center: gws.Point
    context: dict
    items: t.List[gws.MapRenderInputItem]


def start(req: gws.IWebRequest, params: types.Params) -> gws.lib.job.Job:
    job = _create(req, params)
    gws.server.spool.add(job)
    return job


def run(root: gws.IRoot, params: types.Params, project_uid: str, user: gws.IUser):
    base_dir = gws.ensure_dir(gws.PRINT_DIR + '/' + gws.random_string(64))
    w = _Worker(root, '', project_uid, base_dir, params, user)
    return w.run()


def status(job) -> types.StatusResponse:
    return types.StatusResponse(
        jobUid=job.uid,
        state=job.state,
        progress=job.progress,
        steptype=job.steptype or '',
        stepname=job.stepname or '',
        url=gws.action_url_path('printerGetResult', jobUid=job.uid, projectUid=job.project_uid)
    )


##


def _create(req: gws.IWebRequest, params: types.Params) -> gws.lib.job.Job:
    cleanup()

    job_uid = gws.random_string(64)
    base_dir = gws.ensure_dir(gws.PRINT_DIR + '/' + job_uid)

    params_path = base_dir + '/params.pickle'
    gws.serialize_to_path(params, params_path)

    return gws.lib.job.create(
        uid=job_uid,
        user=req.user,
        project_uid=params.projectUid,
        worker=__name__ + '._worker')


def _worker(root: gws.IRoot, job: gws.lib.job.Job):
    job_uid = job.uid
    base_dir = gws.PRINT_DIR + '/' + job_uid

    params_path = base_dir + '/params.pickle'
    params = gws.unserialize_from_path(params_path)

    job.update(state=gws.lib.job.State.running)

    w = _Worker(root, job.uid, job.project_uid, base_dir, params, job.user)
    try:
        w.run()
    except gws.lib.job.PrematureTermination as e:
        gws.log.warn(f'job={job.uid} TERMINATED {e.args!r}')


# remove jobs older than that

_MAX_LIFETIME = 3600 * 1


def cleanup():
    for p in os.listdir(gws.PRINT_DIR):
        d = gws.PRINT_DIR + '/' + p
        age = gws.lib.os2.file_age(d)
        if age > _MAX_LIFETIME:
            gws.lib.os2.run(['rm', '-fr', d])
            gws.lib.job.remove(p)
            gws.log.debug(f'cleaned up job {p} age={age}')


_PAPER_COLOR = 'white'


class _Worker:
    def __init__(self, root: gws.IRoot, job_uid: str, project_uid: str, base_dir: str, params: types.Params, user: gws.IUser):
        self.root = root
        self.base_dir = base_dir
        self.user = user
        self.job_uid = job_uid
        self.project: gws.IProject = self.acquire('gws.base.project', project_uid)
        self.format = params.format or 'pdf'

        self.locale_uid = params.localeUid
        if self.locale_uid not in self.project.locale_uids:
            self.locale_uid = self.project.locale_uids[0]

        self.view_scale = params.scale or 1
        self.view_rotation = params.rotation or 0

        crs = gws.lib.crs.get(params.crs)
        if not crs and self.project:
            crs = self.project.map.crs  # type: ignore
        if not crs:
            raise ValueError('no crs can be found')
        self.view_crs = crs

        self.template: t.Optional[gws.base.template.Object] = None

        self.legends: t.Dict[str, str] = {}
        self.legend_layer_uids = params.legendLayers or []

        if params.type == 'template':
            uid = params.templateUid
            self.template = self.acquire('gws.ext.template', uid)
            if not self.template:
                raise ValueError(f'cannot find template uid={uid!r}')

            # force dpi=OGC_SCREEN_PPI for low-res printing (dpi < OGC_SCREEN_PPI)
            self.view_dpi = max(self.template.dpi_for_quality(params.quality or 0), units.OGC_SCREEN_PPI)
            self.view_size_mm = self.template.map_size

        elif params.type == 'map':
            self.view_dpi = max(gws.to_int(params.dpi), units.OGC_SCREEN_PPI)
            self.view_size_mm = units.point_px2mm((params.mapWidth, params.mapHeight), units.OGC_SCREEN_PPI)

        else:
            raise ValueError(f'invalid print params type: {params.type!r}')

        self.common_render_items = self.prepare_render_items(params.items)

        self.sections: t.List[PreparedSection] = []
        if params.sections:
            self.sections = [self.prepare_section(sec) for sec in params.sections]

        self.default_context = {
            'project': self.project,
            'user': self.user,
            'scale': self.view_scale,
            'rotation': self.view_rotation,
            'localeUid': self.locale_uid,
        }

        nsec = len(self.sections)
        steps = (
                nsec * len(self.common_render_items) +
                sum(len(s.items) for s in self.sections) +
                nsec +
                3)

        self.job_step(steps=steps)

    def run(self):

        section_paths = []

        self.job_step(steptype='begin')

        self.legends = self.render_legends()

        for n, sec in enumerate(self.sections):
            section_paths.append(self.run_section(sec, n))
            self.job_step(steptype='page', stepname=str(n))

        self.job_step(steptype='end')

        comb_path = gws.lib.pdf.concat(section_paths, f'{self.base_dir}/comb.pdf')

        if self.template:
            ctx = gws.merge(self.default_context, self.sections[0].context)
            ctx['page_count'] = gws.lib.pdf.page_count(comb_path)
            res_path = self.template.add_page_elements(
                context=ctx,
                in_path=comb_path,
                out_path=f'{self.base_dir}/res.pdf',
                format='pdf',
            )
        else:
            res_path = comb_path

        if self.format == 'png':
            if self.template:
                size = units.point_mm2px(self.template.page_size, units.PDF_DPI)
            else:
                size = units.point_mm2px(self.view_size_mm, self.view_dpi)
            res_path = gws.lib.pdf.to_image(
                in_path=res_path,
                out_path=res_path + '.png',
                size=size,
                format='png'
            )

        job = self.get_job()
        if job:
            job.update(state=gws.lib.job.State.complete, result={'path': res_path})

        return res_path

    def run_section(self, sec: PreparedSection, n: int):
        renderer = gws.lib.render.Renderer()
        out_path = f'{self.base_dir}/sec-{n}.pdf'

        ri = gws.MapRenderInput(
            items=sec.items + self.common_render_items,
            background_color=_PAPER_COLOR,
            view=gws.lib.render.view_from_center(
                crs=self.view_crs,
                center=sec.center,
                scale=self.view_scale,
                out_size=self.view_size_mm,
                out_size_unit='mm',
                rotation=self.view_rotation,
                dpi=self.view_dpi,
            )
        )

        for item in renderer.run(ri, self.base_dir):
            self.job_step(steptype='layer', stepname=item.layer.title if item.get('layer') else '')

        if self.template:
            tr = self.template.render(
                gws.merge({}, self.default_context, sec.context),
                gws.TemplateRenderArgs(
                    format='pdf',
                    mro=renderer.output,
                    legends=self.legends,
                    out_path=out_path + '.pdf',
                )
            )
            return tr.path

        map_html = gws.lib.render.output_html(renderer.output)

        gws.lib.pdf.render_html(
            '<meta charset="UTF-8"/>' + map_html,
            page_size=self.view_size_mm,
            margin=None,
            out_path=out_path
        )

        return out_path

    def render_legends(self):
        if not self.template or not self.template.legend_mode:
            return {}

        ls = {}

        # @TODO options to print legends even if their layers are hidden
        # @TODO should support printing legends from other maps

        uids = set(self.legend_layer_uids)
        if self.template.legend_layer_uids:
            uids.update(self.template.legend_layer_uids)
            uids = set(s for s in uids if s in self.template.legend_layer_uids)

        for layer_uid in uids:
            layer: gws.ILayer = self.acquire('gws.ext.layer', layer_uid)
            if not layer or not layer.has_legend:
                continue

            lro = layer.render_legend()
            if not lro:
                continue

            if self.template.legend_mode == gws.base.template.LegendMode.image:
                ls[layer.uid] = lro.image_path
            elif self.template.legend_mode == gws.base.template.LegendMode.html:
                ls[layer.uid] = lro.html

        return ls

    def prepare_section(self, sec: types.Section):
        context = sec.get('context', {})
        if self.template and self.template.data_model and context:
            atts = self.template.data_model.apply_to_dict(context)
            context = {a.name: a.value for a in atts}
        return PreparedSection(
            center=sec.center,
            context=context,
            items=self.prepare_render_items(sec.get('items') or []),
        )

    def prepare_render_items(self, items: t.List[types.Item]):

        rs = []

        for n, item in enumerate(items):
            ii = self.prepare_render_item(item)
            if not ii:
                gws.log.warn(f'render item {n} FAILED')
                continue
            rs.append(ii)

        return rs

    def prepare_render_item(self, item: types.Item):
        ii = gws.MapRenderInputItem()

        s = item.get('opacity')
        if s is not None:
            ii.opacity = float(s)
            if ii.opacity == 0:
                return

        # NB: svgs must use the PDF document dpi, not the image dpi!

        s = item.get('style')
        if s:
            s = gws.lib.style.from_props(s)
        if s:
            ii.style = s

        if item.type == 'raster':
            ii.layer = self.acquire('gws.ext.layer', item.layerUid)

            if ii.layer and ii.layer.can_render_box:
                ii.type = gws.MapRenderInputItemType.image_layer
                ii.sub_layers = item.get('subLayers')
                return ii

            return

        if item.type == 'vector':
            ii.layer = self.acquire('gws.ext.layer', item.layerUid)

            if ii.layer and ii.layer.can_render_svg:
                ii.type = gws.MapRenderInputItemType.svg_layer
                ii.dpi = units.PDF_DPI
                return ii

            return

        if item.type == 'bitmap':
            img = self.prepare_bitmap(item)
            if not img:
                return
            ii.type = gws.MapRenderInputItemType.image
            ii.image = img
            return ii

        if item.type == 'url':
            img = self.prepare_bitmap_url(item.url)
            if not img:
                return
            ii.type = gws.MapRenderInputItemType.image
            ii.image = img
            return ii

        if item.type == 'features':
            features = [gws.lib.feature.from_props(p) for p in item.features]
            features = [f for f in features if f and f.shape]
            if not features:
                return
            ii.type = gws.MapRenderInputItemType.features
            ii.features = features
            ii.dpi = units.PDF_DPI
            return ii

        if item.type == 'fragment':
            ii.type = gws.MapRenderInputItemType.fragment
            ii.fragment = gws.SvgFragment(
                points=item.points or [],
                tags=item.tags or [],
                styles=[gws.lib.style.from_props(s) for s in (item.styles or [])])
            ii.dpi = units.PDF_DPI
            return ii

        if not ii.layer:
            return

    def prepare_bitmap(self, item: types.ItemBitmap) -> t.Optional[gws.lib.image.Image]:
        if item.mode in ('RGBA', 'RGB'):
            return gws.lib.image.from_raw_data(item.data, item.mode, (item.width, item.height))

    def prepare_bitmap_url(self, url) -> t.Optional[gws.lib.image.Image]:
        return gws.lib.image.from_data_url(url)

    def acquire(self, klass, uid):
        obj = self.root.find(klass, uid)
        if obj and self.user.can_use(obj):
            return obj

    def get_job(self):
        if not self.job_uid:
            return

        job = gws.lib.job.get(self.job_uid)

        if not job:
            raise gws.lib.job.PrematureTermination('JOB_NOT_FOUND')

        if job.state != gws.lib.job.State.running:
            raise gws.lib.job.PrematureTermination(f'JOB_WRONG_STATE={job.state!r}')

        return job

    def job_step(self, **kwargs):
        job = self.get_job()
        if job:
            job.update(state=gws.lib.job.State.running, step=job.step + 1, **kwargs)
