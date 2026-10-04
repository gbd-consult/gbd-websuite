"""MapServer map wrapper."""

import mapscript
import re

import gws
import gws.lib.image


def version() -> str:
    """Return the MapServer version string.

    Returns:
        The version string, as returned by ``mapscript.msGetVersion``.
    """

    return mapscript.msGetVersion()


class Error(gws.Error):
    """MapServer error."""

    pass


_LAYER_TYPE_TO_MS = {
    gws.MapServerLayerType.point: mapscript.MS_LAYER_POINT,
    gws.MapServerLayerType.line: mapscript.MS_LAYER_LINE,
    gws.MapServerLayerType.polygon: mapscript.MS_LAYER_POLYGON,
    gws.MapServerLayerType.raster: mapscript.MS_LAYER_RASTER,
}



def new_map(config: str = '') -> 'Map':
    """Create a new map.

    Args:
        config: Mapfile content. If empty, an empty map is created.

    Returns:
        A Map object.
    """

    return Map(config)


class Map:
    """Wrapper around a ``mapscript.mapObj``.

    MapServer errors are written to stderr.
    """

    mapObj: mapscript.mapObj
    """The wrapped MapServer map object."""

    def __init__(self, config: str = ''):
        """Create a map.

        Args:
            config: Mapfile content. It is written to a temporary file in the ephemeral directory
                and loaded from there. If empty, an empty map is created.
        """
        if config:
            tmp = gws.c.EPHEMERAL_DIR + '/mapse_' + gws.u.random_string(16) + '.map'
            gws.u.write_file(tmp, config)
            self.mapObj = mapscript.mapObj(tmp)
        else:
            self.mapObj = mapscript.mapObj()

        self.mapObj.setConfigOption('MS_ERRORFILE', 'stderr')

    def copy(self) -> 'Map':
        """Create a copy of the map.

        Returns:
            A new Map object with a clone of the MapServer map.
        """

        c = Map()
        c.mapObj = self.mapObj.clone()
        return c

    def add_layer_from_config(self, config: str) -> mapscript.layerObj:
        """Add a layer to the map from a mapfile ``LAYER`` block.

        Args:
            config: Layer configuration in the mapfile syntax.

        Returns:
            The new MapServer layer object.

        Raises:
            ``Error``: If MapServer cannot create the layer.
        """

        try:
            lo = mapscript.layerObj(self.mapObj)
            lo.updateFromString(config)
            return lo
        except mapscript.MapServerError as exc:
            raise Error(f'ms: add error:: {exc}') from exc

    def add_layer(self, opts: gws.MapServerLayerOptions) -> mapscript.layerObj:
        """Add a layer to the map from layer options.

        The layer is named ``_gws_<n>`` and switched on.

        Args:
            opts: Layer options. ``crs`` is required.

        Returns:
            The new MapServer layer object.

        Raises:
            ``Error``: If the CRS is missing, the connection type is not supported, or MapServer cannot create the layer.
        """

        try:
            lo = self._make_layer(opts)
            return lo
        except mapscript.MapServerError as exc:
            raise Error(f'ms: add error:: {exc}') from exc

    def _make_layer(self, opts: gws.MapServerLayerOptions) -> mapscript.layerObj:
        """Create a MapServer layer from layer options."""
        lo = mapscript.layerObj(self.mapObj)
        lc = self.mapObj.numlayers
        lo.name = f'_gws_{lc}'
        lo.status = mapscript.MS_ON

        if not opts.crs:
            raise Error('missing layer CRS')
        lo.setProjection(opts.crs.epsg)

        if opts.type:
            lo.type = _LAYER_TYPE_TO_MS[opts.type]
        if opts.path:
            lo.data = opts.path
        if opts.tileIndex:
            lo.tileindex = opts.tileIndex
        if opts.processing:
            for p in opts.processing:
                lo.addProcessing(p)
        if opts.transparentColor:
            r, g, b, a = _css_color_to_rgb(opts.transparentColor)
            co = mapscript.colorObj()
            co.setRGB(r, g, b, a)
            lo.offsite = co
        if opts.connectionType:
            if opts.connectionType == 'postgres':
                lo.setConnectionType(mapscript.MS_POSTGIS, '')
            else:
                raise Error(f'unsupported connectionType {opts.connectionType!r}')
        if opts.connectionString:
            lo.connection = opts.connectionString
        if opts.dataString:
            lo.data = opts.dataString
        if opts.sldPath:
            lo.applySLD(gws.u.read_file(opts.sldPath), opts.sldName)

        # @TODO: support style values
        if opts.style:
            cls = mapscript.classObj(lo)

            if opts.style.with_geometry == 'all':
                style_obj = self._create_style_obj(opts.style)
                cls.insertStyle(style_obj)

            if opts.style.with_label == 'all':
                label_obj = self._create_label_obj(opts.style)
                cls.addLabel(label_obj)
                lo.labelitem = 'label'

            if opts.style.marker or opts.style.icon:
                if opts.style.marker:
                    self.mapObj.setSymbolSet('/gws-app/gws/lib/mapserver/symbolset.sym')
                    so = self.style_symbol(opts.style)
                    cls.insertStyle(so)

                if opts.style.icon:
                    symbol = mapscript.symbolObj('icon', opts.style.icon)
                    symbol.type = mapscript.MS_SYMBOL_PIXMAP
                    lo.map.symbolset.appendSymbol(symbol)
                    so = mapscript.styleObj()
                    so.setSymbolByName(lo.map, 'icon')
                    so.size = 100
                    cls.insertStyle(so)
        return lo

    def draw(self, bounds: gws.Bounds, size: gws.Size) -> gws.Image:
        """Render the map as a transparent PNG.

        Args:
            bounds: Extent and CRS to render.
            size: Image size ``(width, height)`` in pixels.

        Returns:
            The rendered image.

        Raises:
            ``Error``: If MapServer cannot render the map.
        """

        # @TODO: options for image format, transparency, etc.

        try:
            gws.debug.time_start(f'mapserver.draw {bounds=} {size=}')

            self.mapObj.setOutputFormat(mapscript.outputFormatObj('AGG/PNG'))
            self.mapObj.outputformat.transparent = mapscript.MS_TRUE

            # setSize re-derives the extent from the current cellsize, so it must
            # come first, otherwise a reused map object overwrites the new extent.

            self.mapObj.setSize(int(size[0]), int(size[1]))
            self.mapObj.setExtent(*bounds.extent)
            self.mapObj.setProjection(bounds.crs.epsg)

            res = self.mapObj.draw()
            img = gws.lib.image.from_bytes(res.getBytes())

            gws.debug.time_end()

            return img

        except mapscript.MapServerError as exc:
            raise Error(f'ms: draw error: {exc}') from exc

    def to_string(self) -> str:
        """Convert the map to a mapfile string.

        Returns:
            Mapfile content.

        Raises:
            ``Error``: If MapServer cannot convert the map.
        """

        try:
            return self.mapObj.convertToString()
        except mapscript.MapServerError as exc:
            raise Error(f'ms: convert error: {exc}') from exc

    def _create_style_obj(self, style: gws.StyleValues) -> mapscript.styleObj:
        """Create a MapServer geometry style from style values."""
        so = mapscript.styleObj()
        if style.fill:
            so.color.setRGB(*_css_color_to_rgb(style.fill))
        if style.stroke:
            so.outlinecolor.setRGB(*_css_color_to_rgb(style.stroke))
            so.outlinewidth = max(0.1 * style.stroke_width, 1)
        if style.stroke_dasharray:
            so.pattern_set(style.stroke_dasharray)
        if style.stroke_dashoffset:
            so.gap = style.stroke_dashoffset
        if style.stroke_linecap:
            so.linecap = _const_mapping.get(style.stroke_linecap.lower())
        if style.stroke_linejoin:
            so.linejoin = _const_mapping.get(style.stroke_linejoin.lower())
        if style.stroke_miterlimit:
            so.linejoinmaxsize = style.stroke_miterlimit
        if style.stroke_width:
            so.width = style.stroke_width
        if style.offset_x:
            so.offsetx = style.offset_x
        if style.offset_y:
            so.offsety = style.offset_y
        return so

    def _create_label_obj(self, style: gws.StyleValues) -> mapscript.labelObj:
        """Create a MapServer label from style values."""
        lo = mapscript.labelObj()
        so = mapscript.styleObj()
        lo.force = mapscript.MS_TRUE

        if style.label_align:
            lo.align = _const_mapping.get(style.label_align)
        if style.label_background:
            so.setGeomTransform('labelpoly')
            so.color.setRGB(*_css_color_to_rgb(style.label_background))
        if style.label_fill:
            lo.color.setRGB(*_css_color_to_rgb(style.label_fill))
        if style.label_font_family:
            lo.font = style.label_font_family  # + '-' + style.label_font_style + '-' + style.label_font_weight
        if style.label_font_size:
            lo.size = style.label_font_size
        if style.label_max_scale:
            lo.maxscaledenom = style.label_max_scale
        if style.label_min_scale:
            lo.minscaledenom = style.label_min_scale
        if style.label_offset_x:
            lo.offsetx = style.label_offset_x
        if style.label_offset_y:
            lo.offsety = style.label_offset_y
        if style.label_padding:
            lo.buffer = max(style.label_padding)
        if style.label_placement:
            lo.position = _const_mapping.get(style.label_placement)
        if style.label_stroke:
            lo.outlinecolor.setRGB(*_css_color_to_rgb(style.label_stroke))
        if style.label_stroke_dasharray:
            so.pattern_set(style.label_stroke_dasharray)
        if style.label_stroke_linecap:
            so.linecap = _const_mapping.get(style.label_stroke_linecap.lower())
        if style.label_stroke_linejoin:
            so.linejoin = _const_mapping.get(style.label_stroke_linejoin.lower())
        if style.label_stroke_miterlimit:
            so.linejoinmaxsize = style.label_stroke_miterlimit
        if style.label_stroke_width:
            lo.outlinewidth = style.label_stroke_width
        lo.insertStyle(so)
        return lo

    def style_symbol(self, style: gws.StyleValues) -> mapscript.styleObj:
        """Create a MapServer marker style from style values.

        The marker symbol is looked up by name in the symbol set of the map.

        Args:
            style: Style values with ``marker`` and optional ``marker_*`` values.

        Returns:
            A MapServer style object.
        """

        mo = self.mapObj
        so = mapscript.styleObj()
        so.setSymbolByName(mo, style.marker)

        if style.marker_fill:
            so.color.setRGB(*_css_color_to_rgb(style.marker_fill))
        if style.marker_size:
            so.size = style.marker_size
        if style.marker_stroke:
            so.outlinecolor.setRGB(*_css_color_to_rgb(style.marker_stroke))
        if style.marker_stroke_dasharray:
            so.pattern_set(style.marker_stroke_dasharray)
        if style.marker_stroke_dashoffset:
            so.gap = style.marker_stroke_dashoffset
        if style.marker_stroke_linecap:
            so.linecap = _const_mapping.get(style.marker_stroke_linecap.lower())
        if style.marker_stroke_linejoin:
            so.linejoin = _const_mapping.get(style.marker_stroke_linejoin.lower())
        if style.marker_stroke_miterlimit:
            so.linejoinmaxsize = style.marker_stroke_miterlimit
        if style.marker_stroke_width:
            so.outlinewidth = style.marker_stroke_width
        return so


def _css_color_to_rgb(s: str) -> tuple[int, int, int, int]:
    """Convert a CSS color (basic name, ``#rrggbb``, ``#rrggbbaa``, ``rgb()`` or ``rgba()``) to RGBA values."""
    s = re.sub(r'\s+', '', s).strip().lower()
    if s in _CSS_COLOR_NAMES:
        r, g, b = _CSS_COLOR_NAMES[s]
        return r, g, b, 255
    m = re.match(r'^#([0-9a-f]{6})$', s)
    if m:
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255
    m = re.match(r'^#([0-9a-f]{8})$', s)
    if m:
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
    m = re.match(r'^rgb\((\d+),(\d+),(\d+)\)$', s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), 255
    m = re.match(r'^rgba\((\d+),(\d+),(\d+),(\d+)\)$', s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    raise ValueError(f'invalid color string: {s!r}')


_CSS_COLOR_NAMES = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'lime': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'aqua': (0, 255, 255),
    'magenta': (255, 0, 255),
    'fuchsia': (255, 0, 255),
    'gray': (128, 128, 128),
    'grey': (128, 128, 128),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'green': (0, 128, 0),
    'purple': (128, 0, 128),
    'teal': (0, 128, 128),
    'navy': (0, 0, 128),
}

_const_mapping = {
    'butt': mapscript.MS_CJC_BUTT,
    'round': mapscript.MS_CJC_ROUND,
    'square': mapscript.MS_CJC_SQUARE,
    'bevel': mapscript.MS_CJC_BEVEL,
    'miter': mapscript.MS_CJC_MITER,
    'left': mapscript.MS_ALIGN_LEFT,
    'center': mapscript.MS_ALIGN_CENTER,
    'right': mapscript.MS_ALIGN_RIGHT,
    'start': mapscript.MS_CL,
    'middle': mapscript.MS_CC,
    'end': mapscript.MS_CR,
}
