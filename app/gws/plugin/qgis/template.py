"""QGIS Print template.

The Qgis print templates work this way:

We read the qgis project and locate a template object within by its title or the index,
by default the first template is taken.

We find all `LayoutHtml` blocks in the template and create our `html` templates from
them, so that they can make use of our placeholders like `@legend`.

When rendering, we render our map as pdf.

Then we render these html templates, and create a clone of the qgis project
with resulting html injected at the proper places.

Then we render the Qgis template without the map, using Qgis `GetPrint` to generate html.

And finally, combine two pdfs so that the qgis pdf is above the map pdf.
This is because we need qgis to draw grids and other decorations above the map.

Caveats/todos:

- both qgis "paper" and the map element must be transparent
- since we create a copy of the qgis project, it must use absolute paths to all assets
- the position of the map in qgis is a couple of mm off when we combine, for better results, the map position/size in qgis must be integer

"""

import gws
import gws.base.template
import gws.plugin.template.html
import gws.lib.htmlx
import gws.lib.mime
import gws.lib.pdf
import gws.gis.render
import gws.types as t

from . import caps, project, provider

gws.ext.new.template('qgis')


class Config(gws.base.template.Config):
    provider: t.Optional[provider.Config]
    """qgis provider"""
    index: t.Optional[int]
    mapPosition: t.Optional[gws.MSize]


class Object(gws.base.template.Object):
    provider: provider.Object
    qgisTemplate: caps.PrintTemplate
    mapPosition: gws.MSize
    htmlTemplates: dict[str, gws.plugin.template.html.Object]

    def configure(self):
        self.provider = provider.get_for(self)
        self._load()

    def render(self, tri):
        # @TODO reload only if changed
        self._load()

        self.notify(tri, 'begin_print')

        # render the map

        self.notify(tri, 'begin_map')
        map_pdf_path = gws.printtemp('q.map.pdf')
        mro = self._render_map(tri, map_pdf_path)
        self.notify(tri, 'end_map')

        # render qgis

        self.notify(tri, 'begin_page')
        qgis_pdf_path = gws.printtemp('q.qgis.pdf')
        self._render_qgis(tri, mro, qgis_pdf_path)
        self.notify(tri, 'end_page')

        if not mro:
            # no map, just return the rendered qgis
            self.notify(tri, 'end_print')
            return gws.ContentResponse(path=qgis_pdf_path)

        # combine map and qgis

        self.notify(tri, 'finalize_print')
        comb_path = gws.printtemp('q.comb.pdf')
        gws.lib.pdf.overlay(map_pdf_path, qgis_pdf_path, comb_path)

        self.notify(tri, 'end_print')
        return gws.ContentResponse(path=comb_path)

    ##

    def _load(self):

        idx = self.cfg('index')
        if idx is not None:
            self.qgisTemplate = self._find_template_by_index(idx)
        elif self.title:
            self.qgisTemplate = self._find_template_by_title(self.title)
        else:
            self.qgisTemplate = self._find_template_by_index(0)

        if not self.title:
            self.title = self.qgisTemplate.title

        self.mapPosition = self.cfg('mapPosition')

        for el in self.qgisTemplate.elements:
            if el.type == 'page' and el.size:
                self.pageSize = el.size
            if el.type == 'map' and el.size:
                self.mapSize = el.size
                self.mapPosition = el.position

        if not self.pageSize or not self.mapSize or not self.mapPosition:
            raise gws.Error('cannot read page or map size')

        self.htmlTemplates = {}

        for el in self.qgisTemplate.elements:
            if el.type != 'html':
                continue
            text = el.attributes.get('html', '')
            uid = 'qgis_html_' + gws.sha256(text)
            self.htmlTemplates[el.uuid] = self.root.create_shared(gws.ext.object.template, uid=uid, type='html', text=text)

    def _find_template_by_index(self, idx):
        try:
            return self.provider.printTemplates[idx]
        except IndexError:
            raise gws.Error(f'print template #{idx} not found')

    def _find_template_by_title(self, title):
        for tpl in self.provider.printTemplates:
            if tpl.title == title:
                return tpl
        raise gws.Error(f'print template {title!r} not found')

    def _render_map(self, tri: gws.TemplateRenderInput, out_path):
        if not tri.maps:
            return

        map0 = tri.maps[0]

        mri = gws.MapRenderInput(
            backgroundColor=map0.backgroundColor,
            bbox=map0.bbox,
            center=map0.center,
            crs=tri.crs,
            dpi=tri.dpi,
            mapSize=self.mapSize,
            planes=map0.planes,
            rotation=map0.rotation,
            scale=map0.scale,
        )

        mro = gws.gis.render.render_map(mri)
        html = gws.gis.render.output_to_html_string(mro)

        x, y, _ = self.mapPosition
        w, h, _ = self.mapSize
        css = f"""
            position: fixed;
            left: {int(x)}mm;
            top: {int(y)}mm;
            width: {int(w)}mm;
            height: {int(h)}mm;
        """
        html = f"<div style='{css}'>{html}</div>"
        gws.lib.htmlx.render_to_pdf(self._decorate_html(html), out_path, self.pageSize)

        return mro

    def _decorate_html(self, html):
        html = '<meta charset="utf8" />\n' + html
        return html

    def _render_qgis(self, tri: gws.TemplateRenderInput, mro: gws.MapRenderOutput, out_path):

        # prepare params for the qgis server

        params = {
            'REQUEST': gws.OwsVerb.GetPrint,
            'CRS': 'EPSG:3857',  # crs doesn't matter, but required
            'FORMAT': 'pdf',
            'TEMPLATE': self.qgisTemplate.title,
            'TRANSPARENT': 'true',
        }

        qgis_project = self.provider.qgis_project()
        changed = self._render_html_blocks(tri, qgis_project)

        if changed:
            # we have html templates, create a copy of the project
            new_project_path = out_path + '.qgs'
            qgis_project.to_path(new_project_path)
            params['MAP'] = new_project_path

        if mro:
            # NB we don't render the map here, but still need map0:xxxx for scale bars and arrows
            # NB the extent is mandatory!
            params = gws.merge(params, {
                'CRS': mro.view.bounds.crs.epsg,
                'MAP0:EXTENT': mro.view.bounds.extent,
                'MAP0:ROTATION': mro.view.rotation,
                'MAP0:SCALE': mro.view.scale,
            })

        res = self.provider.call_server(params)
        gws.write_file_b(out_path, res.content)

    def _render_html_blocks(self, tri: gws.TemplateRenderInput, qgis_project: project.Object):
        if not self.htmlTemplates:
            # there are no html blocks...
            return False

        tri_for_blocks = gws.TemplateRenderInput(
            args=tri.args,
            crs=tri.crs,
            dpi=tri.dpi,
            maps=tri.maps,
            mimeOut=gws.lib.mime.HTML,
            user=tri.user
        )

        html_blocks = {}

        for uuid, tpl in self.htmlTemplates.items():
            res = tpl.render(tri_for_blocks)
            if res.content != tpl.text:
                html_blocks[uuid] = res.content

        if not html_blocks:
            # no blocks are changed - means, they contain no our placeholders
            return False

        for layout_el in qgis_project.xml_root().findall('Layouts/Layout'):
            for item_el in layout_el:
                uuid = item_el.get('uuid')
                if uuid in html_blocks:
                    item_el.set('html', html_blocks[uuid])

        return True
