"""Map render utilities"""

import math

import gws
import gws.lib.extent
import gws.lib.image
import gws.lib.style
import gws.lib.svg
import gws.lib.units as units
import gws.lib.xml3 as xml3
import gws.lib.html2
import gws.types as t

MAX_DPI = 1200
MIN_DPI = units.PDF_DPI


# Map Views

def map_view_from_center(
        size: gws.MSize,
        center: gws.Point,
        crs: gws.ICrs,
        dpi,
        scale,
        rotation=0,
) -> gws.MapView:
    return _map_view(None, center, crs, dpi, rotation, scale, size)


def map_view_from_bbox(
        size: gws.MSize,
        bbox: gws.Extent,
        crs: gws.ICrs,
        dpi,
        rotation=0,
) -> gws.MapView:
    return _map_view(bbox, None, crs, dpi, rotation, None, size)


def _map_view(bbox, center, crs, dpi, rotation, scale, size):
    view = gws.MapView(
        crs=crs,
        dpi=dpi,
        rotation=rotation,
    )

    w, h, u = size
    if u == units.MM:
        view.size_mm = w, h
        view.size_px = units.size_mm_to_px(view.size_mm, view.dpi)
    if u == units.PX:
        view.size_px = w, h
        view.size_mm = units.size_px_to_mm(view.size_px, view.dpi)

    if bbox:
        view.bounds = gws.Bounds(crs=crs, extent=bbox)
        view.center = gws.lib.extent.center(bbox)
        bw, bh = gws.lib.extent.size(bbox)
        view.scale = units.res_to_scale(bw / view.size_px[0])

    elif center:
        view.center = center
        view.scale = scale

        # @TODO assuming projection units are 'm'
        projection_units_per_mm = scale / 1000.0
        size = view.size_mm[0] * projection_units_per_mm, view.size_mm[1] * projection_units_per_mm
        bbox = gws.lib.extent.from_center(center, size)
        view.bounds = gws.Bounds(crs=crs, extent=bbox)

    return view


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

    # (x_map*1000)/scale MM * (dpi/MM_PER_IN) PX/MM => PX
    m2px = (1000.0 / view.scale) * (view.dpi / units.MM_PER_IN)

    ext = view.bounds.extent

    if not view.rotation:
        return translate_int

    ox, oy = translate(*gws.lib.extent.center(ext))
    cosa = math.cos(math.radians(view.rotation))
    sina = math.sin(math.radians(view.rotation))

    return translate_rotate_int


# Rendering


class _Renderer(gws.Data):
    mri: gws.MapRenderInput
    mro: gws.MapRenderOutput
    raster_view: gws.MapView
    vector_view: gws.MapView
    img_count: int
    svg_count: int


def render_map(mri: gws.MapRenderInput, notify: t.Callable = None) -> gws.MapRenderOutput:
    rd = _Renderer(
        mri=mri,
        mro=gws.MapRenderOutput(path=mri.out_path, planes=[]),
        img_count=0,
        svg_count=0
    )

    # vectors always use PDF_DPI
    rd.vector_view = _map_view(mri.bbox, mri.center, mri.crs, units.PDF_DPI, mri.rotation, mri.scale, mri.out_size)
    rd.mro.view = rd.vector_view

    if mri.out_size[2] == units.PX:
        # if they want pixels, use PDF_PDI for rasters as well
        rd.raster_view = rd.vector_view

    elif mri.out_size[2] == units.MM:
        # if they want mm, rasters should use they own dpi
        raster_dpi = min(MAX_DPI, max(MIN_DPI, rd.mri.dpi))
        rd.raster_view = _map_view(mri.bbox, mri.center, mri.crs, raster_dpi, mri.rotation, mri.scale, mri.out_size)

    else:
        raise gws.Error(f'invalid size {mri.out_size!r}')

    # NB: planes are top-to-bottom

    for p in reversed(mri.planes):
        if notify:
            notify('begin_plane', p)
        try:
            _render_plane(rd, p)
        except Exception:
            # swallow exceptions so that we still can render if some layer fails
            gws.log.exception('render: input plane failed')
        if notify:
            notify('end_plane', p)

    return rd.mro


def _render_plane(rd: _Renderer, plane: gws.MapRenderInputPlane):
    s = plane.opacity
    if s is not None:
        opacity = s
    elif plane.layer:
        opacity = plane.layer.opacity
    else:
        opacity = 1

    if plane.type == 'image_layer':
        extra_params = {}
        if plane.sub_layers:
            extra_params = {'layers': plane.sub_layers}
        r = plane.layer.render_box(rd.raster_view, extra_params)
        if r:
            _add_image(rd, gws.lib.image.from_bytes(r), opacity)

    if plane.type == 'image':
        _add_image(rd, plane.image, opacity)
        return

    if plane.type == 'svg_layer':
        els = plane.layer.render_svg_fragment(rd.vector_view, plane.style)
        _add_svg_elements(rd, els, opacity)
        return

    if plane.type == 'features':
        for feature in plane.features:
            els = feature.to_svg_fragment(rd.vector_view, plane.style)
            _add_svg_elements(rd, els, opacity)
        return

    if plane.type == 'svg_soup':
        els = gws.lib.svg.soup_to_fragment(rd.vector_view, plane.soup_points, plane.soup_tags)
        _add_svg_elements(rd, els, opacity)
        return


def _add_image(rd: _Renderer, img, opacity):
    last = rd.mro.planes[-1].type if rd.mro.planes else None

    if last != 'image':
        # NB use background for the first composition only
        background = rd.mri.background_color if rd.img_count == 0 else None
        rd.mro.planes.append(gws.MapRenderOutputPlane(
            type='image',
            image=gws.lib.image.from_size(rd.raster_view.size_px, background)))

    rd.mro.planes[-1].image = rd.mro.planes[-1].image.compose(img, opacity)
    rd.img_count += 1


def _add_svg_elements(rd: _Renderer, elements, opacity):
    # @TODO opacity for svgs

    last = rd.mro.planes[-1].type if rd.mro.planes else None

    if last != 'svg':
        rd.mro.planes.append(gws.MapRenderOutputPlane(
            type='svg',
            elements=[]))

    rd.mro.planes[-1].elements.extend(elements)
    rd.svg_count += 1


# Output


def output_to_html_element(mro: gws.MapRenderOutput, wrap='relative') -> gws.XmlElement:
    w, h = mro.view.size_mm

    css_size = f'left:0;top:0;width:{int(w)}mm;height:{int(h)}mm'
    css_abs = f'position:absolute;{css_size}'

    tags: t.List[gws.XmlElement] = []

    for plane in mro.planes:
        if plane.type == 'image':
            path = mro.path + '.png'
            plane.image.to_path(path)
            tags.append(xml3.tag('img', {'style': css_abs, 'src': path}))
        if plane.type == 'path':
            tags.append(xml3.tag('img', {'style': css_abs, 'src': plane.path}))
        if plane.type == 'svg':
            tags.append(gws.lib.svg.fragment_to_element(plane.elements, {'style': css_abs}))

    css_div = None
    if wrap and wrap in {'relative', 'absolute', 'fixed'}:
        css_div = f'position:{wrap};overflow:hidden;{css_size}'
    return xml3.tag('div', {'style': css_div}, *tags)


def output_to_html_string(mro: gws.MapRenderOutput, wrap='relative') -> str:
    div = output_to_html_element(mro, wrap)
    return xml3.to_string(div)
