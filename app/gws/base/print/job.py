import base64
import os
import pickle
import time

import gws
import gws.types as t
import gws.base.template
import gws.config
import gws.lib.date
import gws.lib.feature
import gws.lib.img
import gws.lib.job
import gws.lib.os2
import gws.lib.pdf
import gws.lib.render
import gws.lib.style
import gws.lib.units as units
import gws.server.spool

from . import types


class PreparedSection(gws.Data):
    center: gws.Point
    context: dict
    items: t.List[gws.MapRenderInputItem]


def start(req: gws.IWebRequest, p: types.Params) -> gws.lib.job.Job:
    job = _create(req, p)
    gws.server.spool.add(job)
    return job


def status(job) -> types.StatusResponse:
    return types.StatusResponse(
        jobUid=job.uid,
        state=job.state,
        progress=job.progress,
        steptype=job.steptype or '',
        stepname=job.stepname or '',
        url=f'{gws.SERVER_ENDPOINT}/printGetResult/jobUid/{job.uid}/projectUid/{job.project_uid}',
    )

##


def _create(req: gws.IWebRequest, p: types.Params) -> gws.lib.job.Job:
    cleanup()

    job_uid = gws.random_string(64)
    base_dir = gws.ensure_dir(gws.PRINT_DIR + '/' + job_uid)

    req_path = base_dir + '/request.pickle'
    gws.write_file_b(req_path, pickle.dumps(p))

    return gws.lib.job.create(
        uid=job_uid,
        user=req.user,
        project_uid=p.projectUid,
        worker=__name__ + '._worker')


def _worker(root: gws.RootObject, job: gws.lib.job.Job):
    job_uid = job.uid
    base_dir = gws.PRINT_DIR + '/' + job_uid

    req_path = base_dir + '/request.pickle'
    params = pickle.loads(gws.read_file_b(req_path))

    job.update(state=gws.lib.job.State.running)

    w = _Worker(root, job, base_dir, params, job.user)
    w.run()


# remove jobs older than that

_MAX_LIFETIME = 3600 * 1


def cleanup():
    for p in os.listdir(gws.PRINT_DIR):
        d = gws.PRINT_DIR + '/' + p
        age = int(time.time() - gws.lib.os2.file_mtime(d))
        if age > _MAX_LIFETIME:
            gws.lib.os2.run(['rm', '-fr', d])
            gws.lib.job.remove(p)
            gws.log.debug(f'cleaned up job {p} age={age}')


_PAPER_COLOR = 'white'


class _Worker:
    def __init__(self, root: gws.RootObject, job: gws.lib.job.Job, base_dir, p: types.Params, user: gws.IUser):
        self.root = root
        self.job = job
        self.base_dir = base_dir
        self.user = user

        self.format = p.format or 'pdf'

        self.project = t.cast(gws.IProject, self.acquire('gws.base.project', job.project_uid))

        self.locale_uid = p.localeUid
        if self.locale_uid not in self.project.locale_uids:
            self.locale_uid = self.project.locale_uids[0]

        self.view_scale = p.scale or 1
        self.view_rotation = p.rotation or 0

        c = p.crs
        if not self.view_crs and self.project:
            c = self.project.map.crs
        if not c:
            raise ValueError('no crs can be found')
        self.view_crs: gws.Crs = c

        self.template: t.Optional[gws.base.template.Object] = None

        self.legend_layer_uids = p.legendLayers or []

        if p.type == 'template':
            self.template = t.cast(gws.base.template.Object, self.acquire('gws.ext.template', p.templateUid))
            if not self.template:
                raise ValueError(f'cannot find template uid={p.templateUid!r}')

            # force dpi=OGC_SCREEN_PPI for low-res printing (dpi < OGC_SCREEN_PPI)
            self.view_dpi = max(self.template.dpi_for_quality(p.quality or 0), units.OGC_SCREEN_PPI)
            self.view_size_mm = self.template.map_size

        elif p.type == 'map':
            self.view_dpi = max(gws.as_int(p.dpi), units.OGC_SCREEN_PPI)
            self.view_size_mm = units.point_px2mm((p.mapWidth, p.mapHeight), units.OGC_SCREEN_PPI)

        else:
            raise ValueError('invalid print params type')

        self.common_render_items = self.prepare_render_items(p.items)

        self.sections: t.List[PreparedSection] = []
        if p.sections:
            self.sections = [self.prepare_section(sec) for sec in p.sections]

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
        try:
            self.run2()
        except gws.lib.job.PrematureTermination as e:
            gws.log.warn(f'job={self.job.uid} TERMINATED {e.args!r}')

    def run2(self):

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
            res_path = self.template.add_headers_and_footers(
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

        self.get_job().update(state=gws.lib.job.State.complete, result={'path': res_path})

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
            return

        ls = {}

        # @TODO options to print legends even if their layers are hidden
        # @TODO should support printing legends from other maps

        uids = set(self.legend_layer_uids)
        if self.template.legend_layer_uids:
            uids.update(self.template.legend_layer_uids)
            uids = set(s for s in uids if s in self.template.legend_layer_uids)

        for layer_uid in uids:
            layer = t.cast(gws.ILayer, self.acquire('gws.ext.layer', layer_uid))
            if not layer or not layer.has_legend:
                continue

            s = None
            if self.template.legend_mode == gws.base.template.LegendMode.image:
                s = layer.render_legend_to_path()
            elif self.template.legend_mode == gws.base.template.LegendMode.html:
                s = layer.render_legend_to_html()
            if s:
                ls[layer.uid] = s

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
            ii.layer = t.cast(gws.ILayer, self.acquire('gws.ext.layer', item.layerUid))

            if ii.layer and ii.layer.can_render_box:
                ii.type = gws.MapRenderInputItemType.image_layer
                ii.sub_layers = item.get('subLayers')
                return ii

            return

        if item.type == 'vector':
            ii.layer = t.cast(gws.ILayer, self.acquire('gws.ext.layer', item.layerUid))

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

    def prepare_bitmap(self, item: types.ItemBitmap):
        if item.mode in ('RGBA', 'RGB'):
            return gws.lib.img.image_api.frombytes(item.mode, (item.width, item.height), item.data)

    def prepare_bitmap_url(self, url):
        data_png = 'data:image/png;base64,'
        if url.startswith(data_png):
            s = url[len(data_png):]
            img = gws.lib.img.image_from_bytes(base64.decodebytes(s.encode('utf8')))
            img.load()
            return img

    def acquire(self, klass, uid):
        obj = self.root.find(klass, uid)
        if obj and self.user.can_use(obj):
            return obj

    def get_job(self):
        job = gws.lib.job.get(self.job.uid)

        if not job:
            raise gws.lib.job.PrematureTermination('NOT_FOUND')

        if job.state != gws.lib.job.State.running:
            raise gws.lib.job.PrematureTermination(f'WRONG_STATE={job.state}')

        return job

    def job_step(self, **kwargs):
        job = self.get_job()
        job.update(state=gws.lib.job.State.running, step=job.step + 1, **kwargs)
