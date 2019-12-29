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
import gws.tools.misc
import gws.tools.pdf
import gws.tools.shell
import gws.tools.units as units

import gws.types as t


class PreparedSection(t.Data):
    center: t.Point
    data: dict
    items: t.List[t.RenderInputItem]


class PrematureTermination(Exception):
    pass


def create(req, params: pt.PrintParams):
    cleanup()

    job_uid = gws.random_string(64)
    base_path = gws.PRINT_DIR + '/' + job_uid
    gws.tools.misc.ensure_dir(base_path)

    req_data = {
        'params': params,
        'user': req.user,
    }

    req_path = base_path + '/request.pickle'
    with open(req_path, 'wb') as fp:
        pickle.dump(req_data, fp)

    return gws.tools.job.create(
        uid=job_uid,
        user_uid=req.user.full_uid,
        worker=__name__ + '._worker',
        args='')


# remove jobs older than that

_lifetime = 3600 * 1


def cleanup():
    for p in os.listdir(gws.PRINT_DIR):
        d = gws.PRINT_DIR + '/' + p
        age = int(time.time() - gws.tools.shell.file_mtime(d))
        if age > _lifetime:
            gws.tools.shell.run(['rm', '-fr', d])
            gws.tools.job.remove(p)
            gws.log.debug(f'cleaned up job {p} age={age}')


def _worker(job):
    job_uid = job.uid
    base_path = gws.PRINT_DIR + '/' + job_uid

    req_path = base_path + '/request.pickle'
    with open(req_path, 'rb') as fp:
        req_data = pickle.load(fp)

    job.update(gws.tools.job.State.running)

    w = _Worker(job_uid, base_path, req_data['params'], req_data['user'])
    w.run()


_PAPER_COLOR = 'white'


class _Worker:
    def __init__(self, job_uid, base_path, p: pt.PrintParams, user: t.User):
        self.job_uid = job_uid
        self.base_path = base_path
        self.p = p
        self.user = user

        self.format = p.get('format') or 'pdf'

        self.project: t.ProjectObject = self.acquire('gws.common.project', self.p.projectUid)

        self.locale = p.get('locale') or ''
        if self.locale not in self.project.locales:
            self.locale = self.project.locales[0]

        self.render_view = t.RenderView()

        self.render_view.scale = self.p.scale
        self.render_view.rotation = self.p.rotation

        if self.p.templateUid:
            self.template: t.TemplateObject = self.acquire('gws.ext.template', self.p.templateUid)
            if not self.template:
                raise ValueError(f'cannot find template uid={self.p.templateUid}')

            # for draft printing (dpi=0), ensure that the client's canvas buffer and our image have the same dimensions
            self.render_view.dpi = self.template.dpi_for_quality(p.get('quality', 0)) or units.OGC_SCREEN_PPI
            self.render_view.size_mm = self.template.map_size
            self.render_view.size_px = units.mm2px_2(self.template.map_size, self.render_view.dpi)
        else:
            # no template uid means we're printing just the map
            # and map pixel dimensions are given in p.mapWidth/p.mapHeight
            # and p.quality == dpi

            self.template = None
            try:
                dpi = min(units.OGC_SCREEN_PPI, int(p.quality))
            except:
                dpi = units.OGC_SCREEN_PPI

            self.render_view.dpi = dpi
            self.render_view.size_mm = units.px2mm_2((p.mapWidth, p.mapHeight), units.OGC_SCREEN_PPI)
            self.render_view.size_px = units.mm2px_2(self.render_view.size_mm, self.render_view.dpi)

        self.common_render_items = self.prepare_render_items(p.items)

        self.sections: t.List[PreparedSection] = [self.prepare_section(sec) for sec in p.sections]

        self.template_vars = {
            'project': self.project,
            'user': self.user,
            'scale': self.render_view.scale,
            'rotation': self.render_view.rotation,
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

        self.check_job(otype='begin')

        for n, sec in enumerate(self.sections):
            section_paths.append(self.run_section(sec, n))
            self.check_job(otype='page', oname=str(n))

        self.check_job(otype='end')

        comb_path = gws.tools.pdf.concat(section_paths, f'{self.base_path}/comb.pdf')

        if self.template:
            ctx = gws.extend(self.sections[0].data, self.template_vars)
            ctx['page_count'] = gws.tools.pdf.page_count(comb_path)
            res_path = self.template.add_headers_and_footers(
                context=ctx,
                in_path=comb_path,
                out_path=f'{self.base_path}/res.pdf',
                format='pdf',
            )
        else:
            res_path = comb_path

        if self.format == 'png':
            res_path = gws.tools.pdf.to_image(
                in_path=res_path,
                out_path=res_path + '.png',
                size=self.render_view.size_px,
                format='png'
            )

        self.get_job().update(gws.tools.job.State.complete, result=res_path)

    def run_section(self, sec: PreparedSection, n: int):
        renderer = gws.gis.render.Renderer()
        out_path = f'{self.base_path}/map-{n}'

        ri = t.RenderInput({
            'items': sec.items + self.common_render_items,
            'background_color': _PAPER_COLOR,
            'view': gws.gis.render.view_from_center(
                center=sec.center,
                scale=self.render_view.scale,
                out_size=self.render_view.size_px,
                out_size_unit='px',
                rotation=self.render_view.rotation,
                dpi=self.render_view.dpi,
            )
        })

        for item in renderer.run(ri, out_path):
            self.check_job(otype='layer', oname=item.layer.title if item.get('layer') else '')

        if self.template:
            tr = self.template.render(
                context=gws.extend(sec.data, self.template_vars),
                render_output=renderer.ro,
                out_path=f'{self.base_path}/sec-{n}.pdf',
                format='pdf',
            )
            return tr.path

        page_size = [
            units.px2mm(self.render_view.size_px[0], self.render_view.dpi),
            units.px2mm(self.render_view.size_px[1], self.render_view.dpi),
        ]

        return gws.tools.pdf.render_html_with_map(
            html='<meta charset="UTF-8"/>@',
            map_placeholder='@',
            page_size=page_size,
            margin=None,
            render_output=renderer.ro,
            out_path=f'{self.base_path}/sec-{n}.pdf',
        )


    def prepare_section(self, sec: pt.PrintSection):
        s = PreparedSection()

        s.center = sec.center
        s.data = {a.name: a.value for a in sec.get('attributes', [])}
        s.items = self.prepare_render_items(sec.get('items') or [])

        return s

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
        ii = t.RenderInputItem()

        s = item.get('opacity')
        if s:
            ii.opacity = float(s)

        s = None
        if item.get('layerUid'):
            s = self.acquire('gws.ext.layer', item.layerUid)
            if not s:
                return
        ii.layer = s

        s = item.get('style')
        if s:
            s = gws.common.style.from_props(s)
        if s:
            ii.style = s

        s = item.get('bitmap')
        if s:
            img = self.prepare_bitmap(s)
            if not img:
                return
            ii.type = t.RenderInputItemType.image
            ii.image = img
            return ii

        s = item.get('features')
        if s:
            features = [gws.gis.feature.from_props(p) for p in item.features]
            features = [f for f in features if f.shape]
            if not features:
                return
            ii.type = t.RenderInputItemType.features
            ii.features = features
            # NB: here and below: svgs must use the PDF document dpi, not the image dpi!
            ii.dpi = units.PDF_DPI
            return ii

        s = item.get('svgFragment')
        if s:
            ii.type = t.RenderInputItemType.fragment
            ii.fragment = s
            ii.dpi = units.PDF_DPI
            return ii

        if not ii.layer:
            return

        if ii.layer.can_render_svg:
            ii.type = t.RenderInputItemType.svg_layer
            ii.dpi = units.PDF_DPI
            return ii

        if ii.layer.can_render_bbox:
            ii.type = t.RenderInputItemType.bbox_layer
            ii.sub_layers = item.get('subLayers')
            return ii

    def prepare_bitmap(self, bitmap: pt.PrintBitmap):
        if bitmap.get('data'):
            if bitmap.mode not in ('RGBA', 'RGB'):
                raise ValueError('invalid bitmap mode')
            return Image.frombytes(bitmap.mode, (bitmap.width, bitmap.height), bitmap.data)

        if bitmap.get('url'):
            s = bitmap.url
            data_png = 'data:image/png;base64,'
            if s.startswith(data_png):
                s = s[len(data_png):]
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
