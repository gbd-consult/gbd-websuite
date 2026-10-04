"""QGIS print template."""
from typing import Optional

import gws
import gws.base.template
import gws.config.util
import gws.plugin.template.html
import gws.lib.htmlx
import gws.lib.osx
import gws.lib.mime
import gws.lib.pdf
import gws.gis.render

from . import caps, project, provider


@gws.ext.config.template('qgis')
class Config(gws.base.template.Config):
    """Print template based on a print layout of a QGIS project."""
    
    provider: Optional[provider.Config]
    """QGIS project that contains the print layout."""
    index: Optional[int]
    """Print layout index in the QGIS project."""
    mapPosition: Optional[gws.UomSizeStr]
    """Position of the main map on the page."""
    cssPath: Optional[gws.FilePath]
    """Stylesheet for the HTML map overlay."""


class _HtmlBlock(gws.Data):
    """A label or HTML item of a print layout, as an ``html`` template."""

    attrName: str
    """Attribute of the layout item that holds the text."""
    template: gws.plugin.template.html.Object
    """Template created from the text."""


@gws.ext.object.template('qgis')
class Object(gws.base.template.Object):
    """Print template based on a print layout of a QGIS project.

    The map is rendered by WebSuite and the layout by QGIS Server; the result
    is one PDF.
    """

    provider: provider.Object
    """QGIS provider."""
    qgisTemplate: caps.PrintTemplate
    """The print layout."""
    mapPosition: gws.UomSize
    """Position of the map item on the page."""
    cssPath: str
    """Stylesheet for the HTML map overlay."""
    htmlBlocks: dict[str, _HtmlBlock]
    """Label and HTML items of the layout, keyed by item UUID."""

    def configure(self):
        self.configure_provider()
        self.cssPath = self.cfg('cssPath', '')
        self._load()

    def configure_provider(self):
        """Set the QGIS provider.

        Returns:
            ``True`` if a provider was set.

        Raises:
            ``gws.Error``: If no provider is found.
        """
        return gws.config.util.configure_provider_for(self, provider.Object)

    def render(self, tri):
        # @TODO reload only if changed
        self._load()

        self.notify(tri, 'begin_print')

        # render the map

        self.notify(tri, 'begin_map')
        map_pdf_path = gws.u.ephemeral_path('q.map.pdf')
        mro = self._render_map(tri, map_pdf_path)
        self.notify(tri, 'end_map')

        # render qgis

        qgis_pdf_path = gws.u.ephemeral_path('q.qgis.pdf')
        self._render_qgis(tri, mro, qgis_pdf_path)

        if not mro:
            # no map, just return the rendered qgis
            self.notify(tri, 'end_print')
            return gws.ContentResponse(contentPath=qgis_pdf_path)

        # combine map and qgis

        self.notify(tri, 'finalize_print')
        comb_path = gws.u.ephemeral_path('q.comb.pdf')
        gws.lib.pdf.overlay(map_pdf_path, qgis_pdf_path, comb_path)

        self.notify(tri, 'end_print')
        return gws.ContentResponse(contentPath=comb_path)

    ##

    def _load(self):
        """Find the print layout and read the page and map sizes.

        The layout is selected by ``index``, or by the template title, or the
        first layout is used.
        """

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

        self._collect_html_blocks()

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
        """Render the first map of the render input as a PDF at the position of the map item."""
        if not tri.maps:
            return

        notify = tri.notify or (lambda *args: None)
        mp = tri.maps[0]

        mri = gws.MapRenderInput(
            backgroundColor=mp.backgroundColor,
            bbox=mp.bbox,
            center=mp.center,
            targetCrs=tri.crs,
            dpi=tri.dpi,
            mapSize=self.mapSize,
            notify=notify,
            planes=mp.planes,
            project=tri.project,
            rotation=mp.rotation,
            scale=mp.scale,
            user=tri.user,
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

        if self.cssPath:
            html = f"""<link rel="stylesheet" href="file://{self.cssPath}">""" + html

        gws.lib.htmlx.render_to_pdf(self._decorate_html(html), out_path, self.pageSize)
        return mro

    def _decorate_html(self, html):
        html = '<meta charset="utf8" />\n' + html
        return html

    def _render_qgis(self, tri: gws.TemplateRenderInput, mro: gws.MapRenderOutput, out_path):
        """Print the layout without the map with QGIS Server GetPrint."""

        # prepare params for the qgis server

        params = {
            'REQUEST': gws.OwsVerb.GetPrint,
            'CRS': 'EPSG:3857',  # crs doesn't matter, but required
            'FORMAT': 'pdf',
            'TEMPLATE': self.qgisTemplate.title,
            'TRANSPARENT': 'true',
            'MAP': self.provider.server_project_path(),
        }

        qgis_project = self.provider.qgis_project()
        changed = self._render_html_blocks(tri, qgis_project)
        project_copy_path = ''

        if changed:
            # we have html templates, create a copy of the project
            # NB it must be in a shared dir, so that qgis container can access it
            # NB since we relocate the project, all assets must be absolute
            project_copy_path = gws.c.QGIS_DIR + '/' + gws.u.random_string(64) + '.qgs'
            qgis_project.to_path(project_copy_path)
            params['MAP'] = project_copy_path

        if mro:
            # NB we don't render the map here, but still need map0:xxxx for scale bars and arrows
            # NB the extent is mandatory!
            params = gws.u.merge(params, {
                'CRS': mro.view.bounds.crs.epsg,
                'MAP0:EXTENT': mro.view.bounds.extent,
                'MAP0:ROTATION': mro.view.rotation,
                'MAP0:SCALE': mro.view.scale,
            })

        res = self.provider.call_server(params)
        gws.u.write_file_b(out_path, res.content)

        if project_copy_path:
            gws.lib.osx.unlink(project_copy_path)

    def _collect_html_blocks(self):
        """Create ``html`` templates from the label and HTML items of the layout."""
        self.htmlBlocks = {}

        for el in self.qgisTemplate.elements:
            if el.type not in {'html', 'label'}:
                continue

            attr = 'html' if el.type == 'html' else 'labelText'
            text = el.attributes.get(attr, '').strip()

            if text:
                self.htmlBlocks[el.uuid] = _HtmlBlock(
                    attrName=attr,
                    template=self.root.create_shared(
                        gws.ext.object.template,
                        uid='qgis_html_' + gws.u.sha256(text),
                        type='html',
                        text=text,
                    )
                )

    def _render_html_blocks(self, tri: gws.TemplateRenderInput, qgis_project: project.Object):
        """Render the HTML blocks into the project XML, return ``True`` if anything changed."""
        if not self.htmlBlocks:
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

        render_results = {}

        for uuid, block in self.htmlBlocks.items():
            res = block.template.render(tri_for_blocks)
            if res.content != block.template.text:
                render_results[uuid] = [block.attrName, res.content]

        if not render_results:
            # no blocks are changed - means, they contain no our placeholders
            return False

        for layout_el in qgis_project.xml_root().findall('Layouts/Layout'):
            for item_el in layout_el:
                uuid = item_el.get('uuid')
                if uuid in render_results:
                    attr, content = render_results[uuid]
                    item_el.set(attr, content)

        return True
