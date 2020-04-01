import io
import os
import time
import base64
import pickle
from PIL import Image

import gws
import gws.common.printer.types as pt
import gws.common.style
import gws.config
import gws.gis.feature
import gws.gis.render
import gws.tools.date
import gws.tools.job
import gws.tools.os2
import gws.tools.pdf
import gws.tools.units as units

import gws.types as t


class PreparedSection(t.Data):
    center: t.Point
    context: dict
    items: t.List[t.MapRenderInputItem]


class PrematureTermination(Exception):
    pass


def create(req: t.IRequest, params: pt.PrintParams) -> gws.tools.job.Job:
    cleanup()

    job_uid = gws.random_string(64)
    base_dir = gws.ensure_dir(gws.PRINT_DIR + '/' + job_uid)

    req_path = base_dir + '/request.pickle'
    with open(req_path, 'wb') as fp:
        pickle.dump(params, fp)

    return gws.tools.job.create(
        uid=job_uid,
        user=req.user,
        worker=__name__ + '._worker')


def _worker(job: gws.tools.job.Job):
    job_uid = job.uid
    base_dir = gws.PRINT_DIR + '/' + job_uid

    req_path = base_dir + '/request.pickle'
    with open(req_path, 'rb') as fp:
        params = pickle.load(fp)

    job.update(gws.tools.job.State.running)

    w = _Worker(job.uid, base_dir, params, job.user)
    w.run()


# remove jobs older than that

_lifetime = 3600 * 1


def cleanup():
    for p in os.listdir(gws.PRINT_DIR):
        d = gws.PRINT_DIR + '/' + p
        age = int(time.time() - gws.tools.os2.file_mtime(d))
        if age > _lifetime:
            gws.tools.os2.run(['rm', '-fr', d])
            gws.tools.job.remove(p)
            gws.log.debug(f'cleaned up job {p} age={age}')


_PAPER_COLOR = 'white'


class _Worker:
    def __init__(self, job_uid, base_dir, p: pt.PrintParams, user: t.IUser):
        self.job_uid = job_uid
        self.base_dir = base_dir
        self.user = user

        self.format = p.format or 'pdf'

        self.project = t.cast(t.IProject, self.acquire('gws.common.project', p.projectUid))

        self.locale = p.locale
        if self.locale not in self.project.locales:
            self.locale = self.project.locales[0]

        self.view_scale = p.scale or 1
        self.view_rotation = p.rotation or 0

        self.view_crs = p.crs
        if not self.view_crs and self.project:
            self.view_crs = self.project.map.crs
        if not self.view_crs:
            raise ValueError('no crs can be found')

        self.template: t.Optional[t.ITemplate] = None

        self.legend_layer_uids = p.legendLayers or []

        if p.type == 'template':
            self.template = t.cast(t.ITemplate, self.acquire('gws.ext.template', p.templateUid))
            if not self.template:
                raise ValueError(f'cannot find template uid={p.templateUid!r}')

            # force dpi=OGC_SCREEN_PPI for low-res printing (dpi < OGC_SCREEN_PPI)
            self.view_dpi = max(self.template.dpi_for_quality(p.quality or 0), units.OGC_SCREEN_PPI)
            self.view_size_mm = self.template.map_size
            self.view_size_px = units.point_mm2px(self.template.map_size, self.view_dpi)

        elif p.type == 'map':
            self.view_dpi = max(gws.as_int(p.dpi), units.OGC_SCREEN_PPI)
            self.view_size_mm = units.point_px2mm((p.mapWidth, p.mapHeight), units.OGC_SCREEN_PPI)
            self.view_size_px = units.point_mm2px(self.view_size_mm, self.view_dpi)

        else:
            raise ValueError('invalid print params type')

        self.common_render_items = self.prepare_render_items(p.items)

        self.sections: t.List[PreparedSection] = [self.prepare_section(sec) for sec in p.sections]

        self.default_context = {
            'project': self.project,
            'user': self.user,
            'scale': self.view_scale,
            'rotation': self.view_rotation,
            'locale': self.locale,
            'lang': self.locale.split('_')[0],
            'date': gws.tools.date.DateFormatter(self.locale),
            'time': gws.tools.date.TimeFormatter(self.locale),
        }

        nsec = len(self.sections)
        steps = (
                nsec * len(self.common_render_items) +
                sum(len(s.items) for s in self.sections) +
                nsec +
                3)

        self.check_job(steps=steps)

    def run(self):
        try:
            self.run2()
        except PrematureTermination as e:
            gws.log.warn(f'job={self.job_uid} TERMINATED {e.args!r}')

    def run2(self):

        section_paths = []

        self.check_job(steptype='begin')

        self.legends = self.render_legends()

        for n, sec in enumerate(self.sections):
            section_paths.append(self.run_section(sec, n))
            self.check_job(steptype='page', stepname=str(n))

        self.check_job(steptype='end')

        comb_path = gws.tools.pdf.concat(section_paths, f'{self.base_dir}/comb.pdf')

        if self.template:
            ctx = gws.merge(self.default_context, self.sections[0].context)
            ctx['page_count'] = gws.tools.pdf.page_count(comb_path)
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
                size = self.view_size_px
            res_path = gws.tools.pdf.to_image(
                in_path=res_path,
                out_path=res_path + '.png',
                size=size,
                format='png'
            )

        self.get_job().update(gws.tools.job.State.complete, result=res_path)

    def run_section(self, sec: PreparedSection, n: int):
        renderer = gws.gis.render.Renderer()
        out_path = f'{self.base_dir}/sec-{n}.pdf'

        ri = t.MapRenderInput(
            items=sec.items + self.common_render_items,
            background_color=_PAPER_COLOR,
            view=gws.gis.render.view_from_center(
                crs=self.view_crs,
                center=sec.center,
                scale=self.view_scale,
                out_size=self.view_size_px,
                out_size_unit='px',
                rotation=self.view_rotation,
                dpi=self.view_dpi,
            )
        )

        for item in renderer.run(ri, self.base_dir):
            self.check_job(steptype='layer', stepname=item.layer.title if item.get('layer') else '')

        if self.template:
            tr = self.template.render(
                context=gws.merge({}, self.default_context, sec.context),
                format='pdf',
                mro=renderer.output,
                legends=self.legends,
                out_path=out_path,
            )
            return tr.path

        page_size = [
            units.px2mm(self.view_size_px[0], self.view_dpi),
            units.px2mm(self.view_size_px[1], self.view_dpi),
        ]

        map_html = gws.gis.render.output_html(renderer.output)

        out_path = gws.tools.pdf.render_html(
            '<meta charset="UTF-8"/>' + map_html,
            page_size=page_size,
            margin=None,
            out_path=out_path
        )

        return out_path

    def render_legends(self):
        if not self.template or not self.template.legend_mode:
            return

        ls = {}

        for layer_uid in self.legend_layer_uids:
            if self.template.legend_layer_uids and layer_uid not in self.template.legend_layer_uids:
                continue

            layer = t.cast(t.ILayer, self.acquire('gws.ext.layer', layer_uid))
            if not layer or not layer.has_legend:
                continue
            s = self.render_legend(layer)
            if s:
                ls[layer.uid] = s

        return ls

    def render_legend(self, layer: t.ILayer):
        if self.template.legend_mode == t.TemplateLegendMode.image:
            return layer.render_legend()

        if self.template.legend_mode == t.TemplateLegendMode.html:
            return layer.render_html_legend()

    def prepare_section(self, sec: pt.PrintSection):
        context = sec.get('context', {})
        if self.template and context:
            context = self.template.normalize_context(context)
        return PreparedSection(
            center=sec.center,
            context=context,
            items=self.prepare_render_items(sec.get('items') or []),
        )

    def prepare_render_items(self, items: t.List[pt.PrintItem]):

        rs = []

        for n, item in enumerate(items):
            ii = self.prepare_render_item(item)
            if not ii:
                gws.log.warn(f'render item {n} FAILED')
                continue
            rs.append(ii)

        return rs

    def prepare_render_item(self, item: pt.PrintItem):
        ii = t.MapRenderInputItem()

        s = item.get('opacity')
        if s is not None:
            ii.opacity = float(s)
            if ii.opacity == 0:
                return

        # NB: svgs must use the PDF document dpi, not the image dpi!

        s = item.get('style')
        if s:
            s = gws.common.style.from_props(s)
        if s:
            ii.style = s
            gws.p(ii.style.values)

        if item.type == 'raster':
            ii.layer = t.cast(t.ILayer, self.acquire('gws.ext.layer', item.layerUid))

            if ii.layer and ii.layer.can_render_box:
                ii.type = t.MapRenderInputItemType.image_layer
                ii.sub_layers = item.get('subLayers')
                return ii

            return

        if item.type == 'vector':
            ii.layer = t.cast(t.ILayer, self.acquire('gws.ext.layer', item.layerUid))

            if ii.layer and ii.layer.can_render_svg:
                ii.type = t.MapRenderInputItemType.svg_layer
                ii.dpi = units.PDF_DPI
                return ii

            return

        if item.type == 'bitmap':
            img = self.prepare_bitmap(item)
            if not img:
                return
            ii.type = t.MapRenderInputItemType.image
            ii.image = img
            return ii

        if item.type == 'url':
            img = self.prepare_bitmap_url(item.url)
            if not img:
                return
            ii.type = t.MapRenderInputItemType.image
            ii.image = img
            return ii

        if item.type == 'features':
            features = [gws.gis.feature.from_props(p) for p in item.features]
            features = [f for f in features if f and f.shape]
            if not features:
                return
            ii.type = t.MapRenderInputItemType.features
            ii.features = features
            ii.dpi = units.PDF_DPI
            return ii

        if item.type == 'fragment':
            ii.type = t.MapRenderInputItemType.fragment
            ii.fragment = item.fragment
            ii.dpi = units.PDF_DPI
            return ii

        if not ii.layer:
            return

    def prepare_bitmap(self, item: pt.PrintItemBitmap):
        if item.mode in ('RGBA', 'RGB'):
            return Image.frombytes(item.mode, (item.width, item.height), item.data)

    def prepare_bitmap_url(self, url):
        data_png = 'data:image/png;base64,'
        if url.startswith(data_png):
            s = url[len(data_png):]
            s = s.encode('utf8')
            s = base64.decodebytes(s)
            img = Image.open(io.BytesIO(s))
            img.load()
            return img

    def acquire(self, klass, uid):
        obj = gws.config.root().find(klass, uid)
        if obj and self.user.can_use(obj):
            return obj

    def get_job(self):
        job = gws.tools.job.get(self.job_uid)

        if not job:
            raise PrematureTermination('NOT_FOUND')

        if job.state != gws.tools.job.State.running:
            raise PrematureTermination(f'WRONG_STATE={job.state}')

        return job

    def check_job(self, **kwargs):
        job = self.get_job()
        job.update(gws.tools.job.State.running, step=job.step + 1, **kwargs)
