import bs4

import gws
import gws.common.template
import gws.config
import gws.gis.feature
import gws.gis.render
import gws.qgis
import gws.tools.misc as misc
import gws.tools.net
import gws.tools.pdf
import gws.tools.mime
import gws.types as t


class Config(t.TemplateConfig):
    """Qgis print template"""
    path: t.filepath


class Object(gws.common.template.Object):
    def __init__(self):
        super().__init__()

        self.title = ''
        self.map_position = [0, 0]
        self.map_size = [0, 0]
        self.page_size = [0, 0]

        self.path = ''
        self.template: gws.qgis.PrintTemplate = None
        self.service: gws.qgis.Service = None

    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.service = gws.qgis.shared_service(self, self.config)
        self.template = self._find_template(self.var('title'))
        if not self.template:
            raise ValueError('print template not found')

        self.uid += '_%d' % self.template.index
        self.title = self.template.title

        self.page_size = self._page_size()
        self.map_size, self.map_position = self._map_size_position()

    def _find_template(self, ref: str):
        pts = self.service.print_templates

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

    def prepare(self, job_uid, context):
        base = gws.PRINT_DIR + '/' + job_uid

        # rewrite the project and replace variables within
        # @TODO fails if there are relative paths in the project
        with open(self.var('path')) as fp:
            prj = fp.read()

        new_prj = _add_variables_to_project(prj, context)

        with open(base + '.qgs', 'wt') as fp:
            fp.write(new_prj)

        context['qgis_temp_path'] = base + '.qgs'
        return context

    def render(self, context, render_output: t.MapRenderOutput = None, out_path=None, format=None):

        # rewrite the project and replace variables within
        # @TODO fails if there are relative paths in the project
        with open(self.path) as fp:
            prj = fp.read()

        temp_prj = _add_variables_to_project(prj, context)
        temp_prj_path = out_path + '.qgs'

        with open(temp_prj_path, 'wt') as fp:
            fp.write(temp_prj)

        # ask qgis to render the template, without the map

        url = 'http://%s:%s' % (
            gws.config.var('server.qgis.host'),
            gws.config.var('server.qgis.port'))

        # NB we don't render the map, but still need map0:xxxx for scale bars to work

        params = {
            'service': 'WMS',
            'version': '1.3',
            'request': 'GetPrint',
            'format': 'pdf',
            'transparent': 'true',
            'template': self.template.title,
            'crs': self.service.supported_crs[0],
            'map': temp_prj_path,
            'map0:scale': render_output.scale,
            'map0:extent': render_output.bbox,
            'map0:rotation': render_output.rotation,
        }

        resp = gws.tools.net.http_request(url, params=params)
        qgis_path = out_path + '_qgis.pdf'
        with open(qgis_path, 'wb') as fp:
            fp.write(resp.content)

        if not render_output:
            return qgis_path

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

        map_path = gws.tools.pdf.render_html_with_map(
            html=map_html,
            map_render_output=render_output,
            map_placeholder='@@@',
            page_size=self.page_size,
            margin=None,
            out_path=out_path + '-map.pdf'
        )

        # merge qgis pdfs + map pdf
        # NB: qgis is ABOVE our map, so the qgis template/map must be transparent!

        out_path = gws.tools.pdf.merge(map_path, qgis_path, out_path)
        return t.TemplateRenderOutput({
            'mimeType': gws.tools.mime.get('pdf'),
            'path': out_path
        })

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
    w, uw = misc.parse_unit(size[0], 'mm')
    h, uh = misc.parse_unit(size[1], 'mm')
    # @TODO inches etc
    return int(w), int(h)


def _add_variables_to_project(prj, context):
    bs = bs4.BeautifulSoup(prj, 'lxml-xml')

    """
    The vars are stored like this in both 2 and 3:
    
    <qgis>
    ....
        <properties>
            ....
            <Variables>
              <variableNames type="QStringList">
                <value>ONE</value>
                <value>TWO</value>
              </variableNames>
              <variableValues type="QStringList">
                <value>11</value>
                <value>22</value>
              </variableValues>
            </Variables>
        </properties>
    </qgis>
    
    """

    props = bs.properties

    if props.Variables:
        vs = dict(zip(
            [str(v.string) for v in props.select('Variables variableNames value')],
            [str(v.string) for v in props.select('Variables variableValues value')],
        ))
        props.Variables.decompose()
    else:
        vs = {}

    vs.update(context)

    props.append(bs.new_tag('Variables'))
    vnames = bs.new_tag('variableNames', type='QStringList')
    vvals = bs.new_tag('variableValues', type='QStringList')

    props.Variables.append(vnames)
    props.Variables.append(vvals)

    for k, v in sorted(vs.items()):
        tag = bs.new_tag('value')
        tag.append(k)
        vnames.append(tag)

        tag = bs.new_tag('value')
        tag.append(str(v))
        vvals.append(tag)

    return str(bs)
