"""QGIS Print template"""

import re

import gws
import gws.types as t
import gws.base.template
import gws.lib.render
import gws.lib.mime
import gws.lib.net
import gws.lib.pdf
import gws.lib.units as units
from . import provider, writer, server


class Config(gws.base.template.Config):
    path: gws.FilePath


class Object(gws.base.template.Object):
    def configure(self):
        

        self.provider = provider.create_shared(self.root, self.config)

        s = self.var('title')
        self.template = self._find_template(s)
        if not self.template:
            raise gws.Error(f'print template {s!r} not found')

        uid = self.var('uid') or '%s_%d' % (gws.sha256(self.provider.path), self.template.index)
        self.set_uid(uid)

        self.source_text = self.provider.source_text
        self._parse()

    def render(self, context: dict, mro=None, out_path=None, legends=None, format=None):

        # rewrite the project and replace variables within
        # @TODO fails if there are relative paths in the project

        if legends and self.legend_mode:
            ctx = {}
            for layer_uid, s in legends.items():
                ctx['GWS_LEGEND_' + gws.sha256(layer_uid)] = s
            ctx['GWS_LEGEND'] = ''.join(legends.values())

            context = gws.merge(ctx, context)

        temp_prj_path = out_path + '.qgs'
        gws.write_file(temp_prj_path, writer.add_variables(self.source_text, context))

        # ask qgis to render the template, without the map
        # NB we still need map0:xxxx for scale bars to work

        params = {
            'service': 'WMS',
            'version': '1.3',
            'request': 'GetPrint',
            'format': 'pdf',
            'transparent': 'true',
            'template': self.template.title,
            'crs': gws.EPSG_3857,  # crs doesn't matter, but required
            'map': temp_prj_path,
        }

        if mro:
            params['map0:scale'] = mro.view.scale
            params['map0:extent'] = mro.view.bounds.extent
            params['map0:rotation'] = mro.view.rotation

        r = server.request(self.root, params)
        qgis_pdf_path = out_path + '_qgis.pdf'
        gws.write_file_b(qgis_pdf_path, r.content)

        if not mro:
            return qgis_pdf_path

        css = ';'.join([
            f'position: absolute',
            f'left:   {self.map_position[0]}mm',
            f'top:    {self.map_position[1]}mm',
            f'width:  {self.map_size[0]}mm',
            f'height: {self.map_size[1]}mm',
        ])

        map_html = gws.lib.render.output_html(mro)
        html = f'<div style="{css}">{map_html}</div>'

        map_path = out_path + '.map.pdf'
        gws.lib.pdf.render_html(
            html,
            page_size=self.page_size,
            margin=None,
            out_path=map_path
        )

        # merge qgis pdfs + map pdf
        # NB: qgis is ABOVE our map, so the qgis template/map must be transparent!

        gws.lib.pdf.merge(map_path, qgis_pdf_path, out_path)

        return gws.TemplateOutput(mime=gws.lib.mime.get('pdf'), path=out_path)

    def add_headers_and_footers(self, context, in_path, out_path, format):
        # @TODO
        return in_path

    def _find_template(self, ref: str):
        pts = self.provider.print_templates

        if not pts:
            return

        if not ref:
            return pts[0]

        if ref.isdigit() and int(ref) < len(pts):
            return pts[int(ref)]

        for tpl in pts:
            if tpl.title == ref:
                return tpl

    def _parse(self):
        self.title = self.template.title
        self.page_size = _page_size(self.template)
        self.map_size, self.map_position = _map_size_position(self.template)

        self.legend_use_all = False
        self.legend_mode = None
        self.legend_layer_uids = []

        # @TODO use a real parser, merge with the html template

        tags_re = r'''(?xs)
            (<|&lt;) (?P<tag1> gws:\w+) (?P<atts1> .*?) / (>|&gt;) 
        '''

        self.source_text = re.sub(tags_re, lambda m: self._parse_tag(m.groupdict()), self.source_text)

        if self.legend_use_all:
            self.legend_layer_uids = []

    def _parse_tag(self, m):
        name = m.get('tag1') or m.get('tag2')
        contents = (m.get('contents2') or '').strip()
        atts = _parse_atts(m.get('atts1') or m.get('atts2') or '')

        if name == 'gws:legend':
            self.legend_mode = gws.TemplateLegendMode.html
            if 'layer' in atts:
                html = ''
                for layer_uid in gws.as_list(atts['layer']):
                    if layer_uid not in self.legend_layer_uids:
                        self.legend_layer_uids.append(layer_uid)
                    html += '[% @GWS_LEGEND_' + gws.sha256(layer_uid) + '%]'
                return html
            self.legend_use_all = True
            return '[% @GWS_LEGEND %]'


def _page_size(tpl):
    # qgis 2:
    if 'paperwidth' in tpl.attrs:
        return (
            float(tpl.attrs['paperwidth']),
            float(tpl.attrs['paperheight'])
        )

    # qgis 3:
    for el in tpl.elements:
        if el.type == 'page':
            return _num_pair(el.attrs['size'])


def _map_size_position(tpl):
    for el in tpl.elements:
        if el.type == 'map':

            # qgis 2:
            if 'pagex' in el.attrs:
                return (
                    (float(el.attrs['width']), float(el.attrs['height'])),
                    (float(el.attrs['pagex']), float(el.attrs['pagey'])),
                )

            # qgis 3:
            if 'position' in el.attrs:
                return (
                    _num_pair(el.attrs['size']),
                    _num_pair(el.attrs['position']),
                )


def _num_pair(s):
    # like '297,210,mm'
    a, b, unit = s.split(',')
    if unit != 'mm':
        # @TODO
        raise ValueError('mm units only please')
    return float(a), float(b)


def _parse_size(size):
    w, uw = gws.lib.units.parse(size[0], default='mm')
    h, uh = gws.lib.units.parse(size[1], default='mm')
    # @TODO inches etc
    return int(w), int(h)


def _parse_atts(a):
    return {k: v.strip() for k, v in re.findall(r'(\w+)=&quot;(.+?)&quot;', a)}
