import os
import time
import base64
import pickle
from PIL import Image

import gws
import gws.config
import gws.types as t
import gws.tools.job
import gws.tools.misc as misc
import gws.tools.pdf
import gws.tools.shell
import gws.common.printer.types as pt
import gws.tools.date
import gws.gis.render
import gws.gis.feature


class PreparedSection(t.Data):
    center: t.Point
    data: dict
    items: t.List[t.MapRenderInputItem]


class PrematureTermination(Exception):
    pass


def create(req, params: pt.PrintParams):
    cleanup()

    job_uid = gws.random_string(64)
    base_path = gws.PRINT_DIR + '/' + job_uid
    misc.ensure_dir(base_path)

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
    def __init__(self, job_uid, base_path, p: pt.PrintParams, user: t.AuthUser):
        self.job_uid = job_uid
        self.base_path = base_path
        self.p = p
        self.user = user

        self.format = p.get('format') or 'pdf'

        self.project: t.ProjectObject = self.acquire('gws.common.project', self.p.projectUid)

        self.locale = p.get('locale') or ''
        if self.locale not in self.project.locales:
            self.locale = self.project.locales[0]

        self.render_input = t.MapRenderInput({
            'scale': self.p.scale,
            'rotation': self.p.rotation,
            'items': [],
            'background_color': _PAPER_COLOR,
        })

        if self.p.templateUid:
            self.template: t.TemplateObject = self.acquire('gws.ext.template', self.p.templateUid)
            if not self.template:
                raise ValueError(f'cannot find template uid={self.p.templateUid}')
            # for draft printing (dpi=0), ensure that the client's canvas buffer and our image have the same dimensions
            self.render_input.dpi = self.template.dpi_for_quality(p.get('quality', 0)) or misc.OGC_SCREEN_PPI
            self.render_input.map_size_px = [
                misc.mm2px(self.template.map_size[0], self.render_input.dpi),
                misc.mm2px(self.template.map_size[1], self.render_input.dpi),
            ]
        else:
            # no template uid means we're printing just the map
            # and map dimensions are given in p.mapWidth/p.mapHeight
            # and p.quality == dpi

            self.template = None
            try:
                q = int(p.quality)
            except:
                q = 0

            if q > misc.OGC_SCREEN_PPI:
                self.render_input.dpi = q

                w = misc.px2mm(p.mapWidth, misc.OGC_SCREEN_PPI)
                h = misc.px2mm(p.mapHeight, misc.OGC_SCREEN_PPI)

                w = misc.mm2px(w, self.render_input.dpi)
                h = misc.mm2px(h, self.render_input.dpi)
            else:
                self.render_input.dpi = misc.OGC_SCREEN_PPI
                w = p.mapWidth
                h = p.mapHeight

            self.render_input.map_size_px = [w, h]

        self.common_render_items = self.prepare_render_items(p.items)

        self.sections: t.List[PreparedSection] = [self.prepare_section(sec) for sec in p.sections]

        self.template_vars = {
            'project': self.project,
            'user': self.user,
            'scale': self.render_input.scale,
            'rotation': self.render_input.rotation,
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
        renderer = gws.gis.render.Renderer()

        self.check_job(otype='begin')

        for n, sec in enumerate(self.sections):

            self.render_input.bbox = self.compute_bbox(sec.center)
            self.render_input.items = sec.items + self.common_render_items
            self.render_input.out_path = f'{self.base_path}/map-{n}'

            for item in renderer.run(self.render_input):
                self.check_job(otype='layer', oname=item.layer.title if item.get('layer') else '')

            self.check_job(otype='page', oname=str(n))

            if self.template:
                tr = self.template.render(
                    context=gws.extend(sec.data, self.template_vars),
                    render_output=renderer.output,
                    out_path=f'{self.base_path}/sec-{n}.pdf',
                    format='pdf',
                )
                section_paths.append(tr.path)
            else:
                page_size = [
                    misc.px2mm(self.render_input.map_size_px[0], self.render_input.dpi),
                    misc.px2mm(self.render_input.map_size_px[1], self.render_input.dpi),
                ]
                path = gws.tools.pdf.render_html_with_map(
                    html='<meta charset="UTF-8"/>@',
                    map_placeholder='@',
                    page_size=page_size,
                    margin=None,
                    map_render_output=renderer.output,
                    out_path=f'{self.base_path}/sec-{n}.pdf',
                )
                section_paths.append(path)

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
                size=self.render_input.map_size_px,
                format='png'
            )

        self.get_job().update(gws.tools.job.State.complete, result=res_path)

    def prepare_section(self, sec: pt.PrintSection):
        s = PreparedSection()

        # NB: user data is supposed to be validated according to the model in frontend
        s.center = sec.center
        s.data = sec.get('data') or {}
        s.items = self.prepare_render_items(sec.get('items') or [])

        return s

    def prepare_render_items(self, items: t.List[pt.PrintItem]):

        render_items = []

        for n, item in enumerate(items):
            ii = t.MapRenderInputItem()

            ii.layer = None

            if item.get('layerUid'):
                ii.layer = self.acquire('gws.ext.layer', item.layerUid)
                if not ii.layer:
                    continue

            if item.get('bitmap'):
                ii.bitmap = self.prepare_bitmap(item, f'{self.base_path}/bitmap-{n}.png')

            if item.get('features'):
                ii.features = []
                for f in item.features:
                    f = gws.gis.feature.Feature(f)
                    if f.shape:
                        ii.features.append(f)

            ii.sub_layers = item.get('subLayers')

            ii.opacity = item.get('opacity') or 1
            ii.style = item.get('style')

            ii.svg_fragment = item.get('svgFragment')

            render_items.append(ii)

        return render_items

    def prepare_bitmap(self, item, out_path):
        bmp = item.bitmap

        if bmp.get('data'):
            if bmp.mode not in ('RGBA', 'RGB'):
                raise ValueError('invalid bitmap mode')
            img = Image.frombytes(bmp.mode, (bmp.width, bmp.height), bmp.data)
            img.save(out_path)
            return out_path

        if bmp.get('url'):
            bdata = bmp.url
            data_png = 'data:image/png;base64,'
            if bdata.startswith(data_png):
                bdata = bdata[len(data_png):]
                bdata = bdata.encode('utf8')
                bdata = base64.decodebytes(bdata)
                with open(out_path, 'wb') as fp:
                    fp.write(bdata)
                return out_path

    def compute_bbox(self, center: t.Point):
        # @TODO assuming projection units are 'm'
        unit_per_mm = self.render_input.scale / 1000.0

        w = misc.px2mm(self.render_input.map_size_px[0], self.render_input.dpi)
        h = misc.px2mm(self.render_input.map_size_px[1], self.render_input.dpi)

        x, y = center

        return [
            x - (w * unit_per_mm) / 2,
            y - (h * unit_per_mm) / 2,
            x + (w * unit_per_mm) / 2,
            y + (h * unit_per_mm) / 2,
        ]

    def acquire(self, klass, uid):
        obj = gws.config.find(klass, uid)
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
