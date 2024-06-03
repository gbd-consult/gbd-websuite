"""Map render utilities"""

import math

import gws
import gws.gis.extent
import gws.lib.image
import gws.lib.svg
import gws.lib.uom
import gws.lib.xmlx as xmlx

MAX_DPI = 1200
MIN_DPI = gws.lib.uom.PDF_DPI


# Map Views

def map_view_from_center(
        size: gws.UomSize,
        center: gws.Point,
        crs: gws.Crs,
        dpi,
        scale,
        rotation=0,
) -> gws.MapView:
    return _map_view(None, center, crs, dpi, rotation, scale, size)


def map_view_from_bbox(
        size: gws.UomSize,
        bbox: gws.Extent,
        crs: gws.Crs,
        dpi,
        rotation=0,
) -> gws.MapView:
    return _map_view(bbox, None, crs, dpi, rotation, None, size)


def _map_view(bbox, center, crs, dpi, rotation, scale, size):
    view = gws.MapView(
        dpi=dpi,
        rotation=rotation,
    )

    w, h, u = size
    if u == gws.Uom.mm:
        view.mmSize = w, h
        view.pxSize = gws.lib.uom.size_mm_to_px(view.mmSize, view.dpi)
    if u == gws.Uom.px:
        view.pxSize = w, h
        view.mmSize = gws.lib.uom.size_px_to_mm(view.pxSize, view.dpi)

    if bbox:
        view.bounds = gws.Bounds(crs=crs, extent=bbox)
        view.center = gws.gis.extent.center(bbox)
        bw, bh = gws.gis.extent.size(bbox)
        view.scale = gws.lib.uom.res_to_scale(bw / view.pxSize[0])
        return view

    if center:
        view.center = center
        view.scale = scale

        # @TODO assuming projection units are 'm'
        projection_units_per_mm = scale / 1000.0
        size = view.mmSize[0] * projection_units_per_mm, view.mmSize[1] * projection_units_per_mm
        bbox = gws.gis.extent.from_center(center, size)
        view.bounds = gws.Bounds(crs=crs, extent=bbox)
        return view

    raise gws.Error('center or bbox required')


def map_view_transformer(view: gws.MapView):
    """Create a pixel transformer f(map_x, map_y) -> (pixel_x, pixel_y) for a view"""

    # @TODO cache the transformer

    def translate(x, y):
        x = x - ext[0]
        y = ext[3] - y
        return x * m2px, y * m2px

    def translate_int(x, y):
        x, y = translate(x, y)
        return int(x), int(y)

    def rotate(x, y):
        return (
            cosa * (x - ox) - sina * (y - oy) + ox,
            sina * (x - ox) + cosa * (y - oy) + oy)

    def translate_rotate_int(x, y):
        x, y = translate(x, y)
        x, y = rotate(x, y)
        return int(x), int(y)

    m2px = 1000.0 * gws.lib.uom.mm_to_px(1 / view.scale, view.dpi)

    ext = view.bounds.extent

    if not view.rotation:
        return translate_int

    ox, oy = translate(*gws.gis.extent.center(ext))
    cosa = math.cos(math.radians(view.rotation))
    sina = math.sin(math.radians(view.rotation))

    return translate_rotate_int


# Rendering


class _Renderer(gws.Data):
    mri: gws.MapRenderInput
    mro: gws.MapRenderOutput
    rasterView: gws.MapView
    vectorView: gws.MapView
    imgCount: int
    svgCount: int


def render_map(mri: gws.MapRenderInput) -> gws.MapRenderOutput:
    rd = _Renderer(
        mri=mri,
        mro=gws.MapRenderOutput(planes=[]),
        imgCount=0,
        svgCount=0
    )

    # vectors always use PDF_DPI
    rd.vectorView = _map_view(mri.bbox, mri.center, mri.crs, gws.lib.uom.PDF_DPI, mri.rotation, mri.scale, mri.mapSize)

    if mri.mapSize[2] == gws.Uom.px:
        # if they want pixels, use PDF_PDI for rasters as well
        rd.rasterView = rd.vectorView

    elif mri.mapSize[2] == gws.Uom.mm:
        # if they want mm, rasters should use they own dpi
        raster_dpi = min(MAX_DPI, max(MIN_DPI, rd.mri.dpi))
        rd.rasterView = _map_view(mri.bbox, mri.center, mri.crs, raster_dpi, mri.rotation, mri.scale, mri.mapSize)

    else:
        raise gws.Error(f'invalid size {mri.mapSize!r}')

    # NB: planes are top-to-bottom

    for n, p in enumerate(reversed(mri.planes)):
        if mri.notify:
            mri.notify('begin_plane', p)
        try:
            _render_plane(rd, p)
        except Exception:
            gws.log.exception(f'RENDER_FAILED: plane {len(mri.planes) - n - 1}')
        if mri.notify:
            mri.notify('end_plane', p)

    rd.mro.view = rd.vectorView
    return rd.mro


def _render_plane(rd: _Renderer, plane: gws.MapRenderInputPlane):
    s = plane.opacity
    if s is not None:
        opacity = s
    elif plane.layer:
        opacity = plane.layer.opacity
    else:
        opacity = 1

    if plane.type == gws.MapRenderInputPlaneType.imageLayer:
        extra_params = {}
        if plane.subLayers:
            extra_params = {'layers': plane.subLayers}
        lro = plane.layer.render(gws.LayerRenderInput(
            type=gws.LayerRenderInputType.box,
            view=rd.rasterView,
            extraParams=extra_params,
            user=rd.mri.user,
        ))
        if lro:
            _add_image(rd, gws.lib.image.from_bytes(lro.content), opacity)
        return

    if plane.type == gws.MapRenderInputPlaneType.image:
        _add_image(rd, plane.image, opacity)
        return

    if plane.type == gws.MapRenderInputPlaneType.svgLayer:
        lro = plane.layer.render(gws.LayerRenderInput(
            type=gws.LayerRenderInputType.svg,
            view=rd.vectorView,
            style=plane.styles[0] if plane.styles else None,
            user=rd.mri.user,
        ))
        if lro:
            _add_svg_elements(rd, lro.tags, opacity)
        return

    if plane.type == gws.MapRenderInputPlaneType.features:
        style_dct = {}
        if plane.styles:
            style_dct = {s.cssSelector: s for s in plane.styles}
        for f in plane.features:
            tags = f.to_svg(rd.vectorView, f.views.get('label', ''), style_dct.get(f.cssSelector))
            _add_svg_elements(rd, tags, opacity)
        return

    if plane.type == gws.MapRenderInputPlaneType.svgSoup:
        els = gws.lib.svg.soup_to_fragment(rd.vectorView, plane.soupPoints, plane.soupTags)
        _add_svg_elements(rd, els, opacity)
        return


def _add_image(rd: _Renderer, img, opacity):
    last_type = rd.mro.planes[-1].type if rd.mro.planes else None

    if last_type != gws.MapRenderOutputPlaneType.image:
        # NB use background for the first composition only
        background = rd.mri.backgroundColor if rd.imgCount == 0 else None
        rd.mro.planes.append(gws.MapRenderOutputPlane(
            type=gws.MapRenderOutputPlaneType.image,
            image=gws.lib.image.from_size(rd.rasterView.pxSize, background)))

    rd.mro.planes[-1].image = rd.mro.planes[-1].image.compose(img, opacity)
    rd.imgCount += 1


def _add_svg_elements(rd: _Renderer, elements, opacity):
    # @TODO opacity for svgs

    last_type = rd.mro.planes[-1].type if rd.mro.planes else None

    if last_type != gws.MapRenderOutputPlaneType.svg:
        rd.mro.planes.append(gws.MapRenderOutputPlane(
            type=gws.MapRenderOutputPlaneType.svg,
            elements=[]))

    rd.mro.planes[-1].elements.extend(elements)
    rd.svgCount += 1


# Output


def output_to_html_element(mro: gws.MapRenderOutput, wrap='relative') -> gws.XmlElement:
    w, h = mro.view.mmSize

    css_size = f'left:0;top:0;width:{int(w)}mm;height:{int(h)}mm'
    css_abs = f'position:absolute;{css_size}'

    tags: list[gws.XmlElement] = []

    for plane in mro.planes:
        if plane.type == gws.MapRenderOutputPlaneType.image:
            img_path = plane.image.to_path(gws.u.printtemp('mro.png'))
            tags.append(xmlx.tag('img', {'style': css_abs, 'src': img_path}))
        if plane.type == gws.MapRenderOutputPlaneType.path:
            tags.append(xmlx.tag('img', {'style': css_abs, 'src': plane.path}))
        if plane.type == gws.MapRenderOutputPlaneType.svg:
            tags.append(gws.lib.svg.fragment_to_element(plane.elements, {'style': css_abs}))

    if not tags:
        tags.append(xmlx.tag('img'))

    css_div = None
    if wrap and wrap in {'relative', 'absolute', 'fixed'}:
        css_div = f'position:{wrap};overflow:hidden;{css_size}'
    return xmlx.tag('div', {'style': css_div}, *tags)


def output_to_html_string(mro: gws.MapRenderOutput, wrap='relative') -> str:
    div = output_to_html_element(mro, wrap)
    return div.to_string()
