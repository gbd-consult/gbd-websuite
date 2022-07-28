"""QGIS Print template"""

import gws
import gws.base.template
import gws.lib.html2
import gws.lib.mime
import gws.gis.ows
import gws.lib.pdf
import gws.gis.render
import gws.lib.xml2 as xml2
import gws.types as t

from . import provider, caps

_dummy_fn = lambda *args: None


@gws.ext.config.template('qgis')
class Config(gws.base.template.Config):
    path: gws.FilePath
    index: t.Optional[int]
    mapPosition: t.Optional[gws.MSize]


@gws.ext.object.template('qgis')
class Object(gws.base.template.Object):
    provider: provider.Object
    template: caps.PrintTemplate
    source_text: str
    map_position: gws.MSize

    def configure(self):
        if self.var('_provider'):
            self.provider = self.var('_provider')
        else:
            self.provider = self.root.create_object(provider.Object, self.config, shared=True)

        s = self.var('title') or self.var('index')
        self.template = self.provider.print_template(s)
        if not self.template:
            raise gws.Error(f'print template {s!r} not found')

        uid = self.var('uid') or (gws.sha256(self.provider.path) + '_' + str(self.template.index))
        self.set_uid(uid)

        self.map_position = self.var('mapPosition')

        for el in self.template.elements:
            if el.type == 'page' and el.size:
                self.page_size = el.size
            if el.type == 'map' and el.size:
                self.map_size = el.size
                self.map_position = el.position

        if not self.page_size or not self.map_size or not self.map_position:
            raise gws.Error('cannot read page or map size')

    def render(self, tri, notify=None):
        notify = notify or _dummy_fn

        notify('begin_print')

        notify('begin_map')
        map_pdf_path = gws.tempname('q.map.pdf')
        mro = self._render_map(tri, notify, map_pdf_path)
        notify('end_map')

        notify('begin_page')
        qgis_pdf_path = gws.tempname('q.qgis.pdf')
        self._render_qgis(tri, notify, mro, qgis_pdf_path)
        notify('end_page')

        if not mro:
            notify('end_print')
            return gws.ContentResponse(path=qgis_pdf_path)

        # merge qgis pdfs + map pdf
        # NB: qgis is ABOVE our map, so the qgis template and map must be transparent!
        # (this is because we need qgis to draw grids above the map)
        # @TODO automatic transparency

        notify('finalize_print')
        comb_path = gws.tempname('q.comb.pdf')
        gws.lib.pdf.overlay(map_pdf_path, qgis_pdf_path, comb_path)

        notify('end_print')
        return gws.ContentResponse(path=comb_path)

    def _render_map(self, tri: gws.TemplateRenderInput, notify, out_path):
        if not tri.maps:
            return

        mp = tri.maps[0]

        mri = gws.MapRenderInput(
            background_color=mp.background_color,
            bbox=mp.bbox,
            center=mp.center,
            crs=tri.crs,
            dpi=tri.dpi,
            out_size=self.map_size,
            planes=mp.planes,
            rotation=mp.rotation,
            scale=mp.scale,
        )

        mro = gws.gis.render.render_map(mri, notify)
        html = gws.gis.render.output_to_html_string(mro)

        # create an empty page with the map positioned at the right place
        # @TODO the position is a couple mm off
        # for better results, the map pos/size in qgis must be integer

        x, y, _ = self.map_position
        w, h, _ = self.map_size
        css = f"""
            position: fixed;
            left: {int(x)}mm;
            top: {int(y)}mm;
            width: {int(w)}mm;
            height: {int(h)}mm;
        """
        html = f"<div style='{css}'>{html}</div>"
        gws.lib.html2.render_to_pdf(html, out_path, self.page_size)

        return mro

    def _render_qgis(self, tri: gws.TemplateRenderInput, notify, mro: gws.MapRenderOutput, out_path):
        # filter html blocks in the template through our html templating
        # NB we only process `LayoutHtml` blocks, not `LayoutLabel` (see caps),
        # because the latter has problems with images as of 3.10
        # once html blocks are filtered, we create a copy of the qgs file with substituted html blocks
        # @TODO relative paths should be corrected

        project_path = self.provider.path
        boxes = self._render_html_boxes(tri)
        if boxes:
            project_path = gws.tempname('q.boxes.qgs')
            self._inject_html_boxes(tri, boxes, project_path)

        return self._render_qgis_to_pdf(tri, mro, project_path, out_path)

    def _render_html_boxes(self, tri):
        # iterate the template XML tree, find `LayoutHtml` blocks
        # and create a dict block.uuid -> rendered_html

        tri_for_boxes = gws.TemplateRenderInput(
            context=tri.context,
            crs=tri.crs,
            dpi=tri.dpi,
            locale_uid=tri.locale_uid,
            maps=tri.maps,
            out_mime=gws.lib.mime.HTML,
            user=tri.user
        )

        boxes = {}

        for el in self.template.elements:
            if el.type != 'html':
                continue
            text = el.attributes.get('html', '')
            tpl = self.root.create_object('gws.ext.template.html', gws.Config(text=text))
            res = tpl.render(tri_for_boxes)
            if res.content != text:
                boxes[el.uuid] = res.content

        return boxes

    def _inject_html_boxes(self, tri, boxes, project_path):
        # iterate the template XML tree and inject our boxes
        # see `caps.py` for the print layout structure

        root_el = xml2.from_path(self.provider.path)
        for layout_el in xml2.all(root_el, 'Layouts Layout'):
            for item_el in layout_el.children:
                uuid = item_el.attributes.get('uuid')
                if uuid in boxes:
                    item_el.attributes['html'] = boxes[uuid]

        gws.write_file(project_path, xml2.to_string(root_el))

    def _render_qgis_to_pdf(self, tri, mro, project_path, out_path):
        # invoke the qgis server and make them render a pdf from the template (without a map)

        params = {
            'version': '1.3.0',
            'format': 'pdf',
            'transparent': 'true',
            'template': self.template.title,
            'crs': 'EPSG:3857',  # crs doesn't matter, but required
            'map': project_path,
        }

        # NB we still need map0:xxxx for scale bars to work
        # extent is mandatory!

        if mro:
            params['map0:scale'] = mro.view.scale
            params['map0:extent'] = mro.view.bounds.extent
            params['map0:rotation'] = mro.view.rotation
            params['crs'] = mro.view.bounds.crs.epsg

        res = gws.gis.ows.request.get(
            self.provider.url,
            gws.OwsProtocol.WMS,
            gws.OwsVerb.GetPrint,
            params=params,
        )

        gws.write_file_b(out_path, res.content)
