"""QGIS Print template"""

import bs4

import gws
import gws.common.template
import gws.tools.mime
import gws.tools.misc
import gws.tools.units as units
import gws.gis.render
import gws.tools.net
import gws.tools.pdf

import gws.types as t

from . import types, provider, writer, server


class Config(gws.common.template.Config):
    path: t.FilePath


class Object(gws.common.template.Object):
    def __init__(self):
        super().__init__()

        self.title = ''
        self.map_position = [0, 0]
        self.map_size = [0, 0]
        self.page_size = [0, 0]

        self.path = ''
        self.template: types.PrintTemplate = None
        self.provider: provider.Object = None

    def configure(self):
        super().configure()

        self.provider = provider.create_shared(self, self.config)
        self.template = self._find_template(self.var('title'))
        if not self.template:
            raise gws.Error('print template not found')

        uid = self.var('uid') or '%s_%d' % (gws.tools.misc.sha256(self.path), self.template.index)
        self.set_uid(uid)

        self.title = self.template.title
        self.page_size = self._page_size()
        self.map_size, self.map_position = self._map_size_position()

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

    def _page_size(self):
        # qgis 2:
        if 'paperwidth' in self.template.attrs:
            return [
                float(self.template.attrs['paperwidth']),
                float(self.template.attrs['paperheight'])
            ]

        # qgis 3:
        for el in self.template.elements:
            if el.type == 'page':
                return _num_pair(el.attrs['size'])

    def _map_size_position(self):
        for el in self.template.elements:
            if el.type == 'map':

                # qgis 2:
                if 'pagex' in el.attrs:
                    return (
                        [float(el.attrs['width']), float(el.attrs['height'])],
                        [float(el.attrs['pagex']), float(el.attrs['pagey'])],
                    )

                # qgis 3:
                if 'position' in el.attrs:
                    return (
                        _num_pair(el.attrs['size']),
                        _num_pair(el.attrs['position']),
                    )

    def render(self, context, render_output: t.RenderOutput = None, out_path=None, format=None):

        # rewrite the project and replace variables within
        # @TODO fails if there are relative paths in the project

        temp_prj_path = out_path + '.qgs'
        with open(temp_prj_path, 'wt') as fp:
            fp.write(writer.add_variables(self.path, context))

        # ask qgis to render the template, without the map
        # NB we still need map0:xxxx for scale bars to work

        resp = server.request(self.root, {
            'service': 'WMS',
            'version': '1.3',
            'request': 'GetPrint',
            'format': 'pdf',
            'transparent': 'true',
            'template': self.template.title,
            'crs': gws.EPSG_3857,  # crs doesn't matter, but required
            'map': temp_prj_path,
            'map0:scale': render_output.view.scale,
            'map0:extent': render_output.view.bounds.extent,
            'map0:rotation': render_output.view.rotation,
        })

        qgis_pdf_path = out_path + '_qgis.pdf'
        with open(qgis_pdf_path, 'wb') as fp:
            fp.write(resp.content)

        if not render_output:
            return qgis_pdf_path

        # create a temp html for the map

        css = ';'.join([
            f'position: absolute',
            f'left:   {self.map_position[0]}mm',
            f'top:    {self.map_position[1]}mm',
            f'width:  {self.map_size[0]}mm',
            f'height: {self.map_size[1]}mm',
        ])

        map_html = f'<meta charset="utf8"/><div style="{css}">@@@</div>'

        # render the map html into a pdf

        html = gws.gis.render.create_html_with_map(
            html=map_html,
            render_output=render_output,
            map_placeholder='@@@',
            page_size=self.page_size,
            margin=None,
            out_path=out_path + '-map.pdf'
        )

        map_path = gws.tools.pdf.render_html(
            html,
            page_size=self.page_size,
            margin=None,
            out_path=out_path
        )

        # merge qgis pdfs + map pdf
        # NB: qgis is ABOVE our map, so the qgis template/map must be transparent!

        out_path = gws.tools.pdf.merge(map_path, qgis_pdf_path, out_path)

        return t.TemplateOutput(mime=gws.tools.mime.get('pdf'), path=out_path)

    def add_headers_and_footers(self, context, in_path, out_path, format):
        # @TODO
        return in_path


def _num_pair(s):
    # like '297,210,mm'
    a, b, unit = s.split(',')
    if unit != 'mm':
        # @TODO
        raise ValueError('mm units only please')
    return [float(a), float(b)]


def _parse_size(size):
    w, uw = gws.tools.units.parse(size[0], 'mm')
    h, uh = gws.tools.units.parse(size[1], 'mm')
    # @TODO inches etc
    return int(w), int(h)
