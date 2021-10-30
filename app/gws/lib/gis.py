"""Miscellaneous GIS-related utilities"""

import re

import gws
import gws.lib.extent
import gws.lib.metadata
import gws.lib.proj
import gws.types as t


class PreparedWmsSearch(gws.Data):
    params: dict
    request_crs: gws.Crs
    axis: gws.Axis


def prepare_wms_search(
        shape: gws.IShape,
        protocol_version,
        force_crs: t.Optional[gws.Crs] = None,
        supported_crs: t.Optional[t.List[gws.Crs]] = None,
        invert_axis_crs: t.Optional[t.List[gws.Crs]] = None
) -> t.Optional[PreparedWmsSearch]:
    if not shape:
        return None

    if shape.geometry_type != gws.GeometryType.point:
        return None

    request_crs = force_crs or best_crs(shape.crs, supported_crs)
    axis = best_axis(request_crs, gws.OwsProtocol.WMS, protocol_version, invert_axis_crs)
    shape = shape.transformed_to(request_crs)

    box_size_m = 500
    box_size_px = 500

    bbox = (
        shape.x - (box_size_m >> 1),
        shape.y - (box_size_m >> 1),
        shape.x + (box_size_m >> 1),
        shape.y + (box_size_m >> 1),
    )

    if axis == gws.AXIS_YX:
        bbox = gws.lib.extent.swap_xy(bbox)

    v3 = protocol_version >= '1.3'

    params = {
        'BBOX': bbox,
        'WIDTH': box_size_px,
        'HEIGHT': box_size_px,
        'I' if v3 else 'X': box_size_px >> 1,
        'J' if v3 else 'Y': box_size_px >> 1,
        'CRS' if v3 else 'SRS': request_crs,
        'VERSION': protocol_version,
    }

    return PreparedWmsSearch(params=params, request_crs=request_crs, axis=axis)


def best_axis(crs: gws.Crs, protocol: gws.OwsProtocol, protocol_version, invert_axis_crs: t.Optional[t.List[gws.Crs]] = None) -> gws.Axis:
    # inverted_axis_crs_list - list of EPSG crs'es which are known to have an inverted axis for this service
    # crs - crs we're going to use with the service

    proj = gws.lib.proj.to_proj(crs)
    if invert_axis_crs and proj.epsg in invert_axis_crs:
        return gws.AXIS_YX

    # @TODO some logic to guess the axis, based on crs, service protocol and version
    # see https://docs.geoserver.org/latest/en/user/services/wfs/basics.html#wfs-basics-axis
    return gws.AXIS_XY


def best_crs(target_crs: gws.Crs, supported_crs: t.Optional[t.List[gws.Crs]]) -> gws.Crs:
    # find the best matching crs for the target crs and the list of supported crs.

    if not supported_crs:
        return target_crs

    # if target_crs is in the list, we're fine

    crs = gws.lib.proj.find(target_crs, supported_crs)
    if crs:
        return crs

    # @TODO find a projection with less errors

    # if webmercator is supported, use it

    crs = gws.lib.proj.find(gws.EPSG_3857, supported_crs)
    if crs:
        gws.log.debug(f'best_crs: using {crs!r} for {target_crs!r}')
        return crs

    # return first non-geographic CRS

    for crs in supported_crs:
        p = gws.lib.proj.to_proj(crs)
        if p and not p.is_geographic:
            gws.log.debug(f'best_crs: using {p.epsg!r} for {target_crs!r}')
            return p.epsg

    raise gws.Error(f'no match for crs={target_crs!r} in {supported_crs!r}')


class SourceStyle(gws.Data):
    is_default: bool
    legend_url: gws.Url
    metadata: gws.lib.metadata.Metadata
    name: str


class TileMatrix(gws.Data):
    uid: str
    scale: float
    x: float
    y: float
    width: float
    height: float
    tile_width: float
    tile_height: float
    extent: gws.Extent


class TileMatrixSet(gws.Data):
    uid: str
    crs: gws.Crs
    matrices: t.List[TileMatrix]


class SourceLayer(gws.Data):
    a_level: int
    a_path: str
    a_uid: str

    data_source: dict

    supported_crs: t.List[gws.Crs]
    supported_bounds: t.List[gws.Bounds]

    is_expanded: bool
    is_group: bool
    is_image: bool
    is_queryable: bool
    is_visible: bool

    layers: t.List['SourceLayer']

    metadata: gws.lib.metadata.Metadata
    name: str
    title: str
    format: str

    opacity: int
    scale_range: t.List[float]
    styles: t.List[SourceStyle]
    legend_url: gws.Url
    resource_urls: dict

    matrix_sets: t.List[TileMatrixSet]
    matrix_ids: t.List[str]


class SourceLayerFilter(gws.Data):
    """Source layer filter"""

    level: int = 0  #: match only layers at this level
    names: t.Optional[t.List[str]]  #: match these layer names (top-to-bottom order)
    pattern: gws.Regex = ''  #: match layers whose full path matches a pattern


def source_layer_matches(sl: SourceLayer, slf: SourceLayerFilter) -> bool:
    """Check if a source layer matches the filter"""

    if not slf:
        return True

    s = gws.get(slf, 'level')
    if s and sl.a_level != s:
        return False

    s = gws.get(slf, 'names')
    if s is not None and sl.name not in s:
        return False

    s = gws.get(slf, 'pattern')
    if s and not re.search(s, sl.a_path):
        return False

    return True


def filter_source_layers(layers: t.List[SourceLayer], slf: SourceLayerFilter) -> t.List[SourceLayer]:
    """Filter source layers by the given layer filter."""

    if slf:
        s = gws.get(slf, 'level')
        if s:
            layers = [sl for sl in layers if sl.a_level == s]

        s = gws.get(slf, 'names')
        if s:
            # NB: if 'names' is given, maintain the given order, which is expected to be top-to-bottom
            # see note in ext/layers/wms
            layers2 = []
            for name in s:
                for sl in layers:
                    if sl.name == name:
                        layers2.append(sl)
                        break
            layers = layers2

        s = gws.get(slf, 'pattern')
        if s:
            layers = [sl for sl in layers if re.search(s, sl.a_path)]

    return layers


def enum_source_layers(layers: t.List[SourceLayer], is_image=False, is_queryable=False) -> t.List[SourceLayer]:
    found = []

    for sl in layers:
        if is_image and sl.is_image:
            found.append(sl)
        elif is_queryable and sl.is_queryable:
            found.append(sl)
        else:
            found.extend(enum_source_layers(sl.layers, is_image, is_queryable))

    return found


def bounds_from_source_layers(source_layers: t.List[SourceLayer], target_crs) -> gws.Bounds:
    """Return merged bounds from a list of source layers in the target_crs."""

    exts = []

    for sl in source_layers:
        if not sl.supported_bounds:
            continue
        bb = _best_bounds(sl.supported_bounds, target_crs)
        if bb:
            e = gws.lib.extent.transform(bb.extent, bb.crs, target_crs)
            exts.append(e)

    if exts:
        return gws.Bounds(crs=target_crs, extent=gws.lib.extent.merge(exts))


def crs_from_source_layers(source_layers: t.List[SourceLayer]) -> t.List[gws.Crs]:
    """Return an intersection of crs supported by each source layer."""

    cs: t.Set[str] = set()

    for sl in source_layers:
        if not sl.supported_crs:
            continue
        if not cs:
            cs.update(sl.supported_crs)
        else:
            cs = cs.intersection(sl.supported_crs)

    return sorted(cs)


def _best_bounds(bs: t.List[gws.Bounds], target_crs):
    for b in bs:
        if gws.lib.proj.equal(b.crs, target_crs):
            return b
    for b in bs:
        if gws.lib.proj.equal(b.crs, gws.EPSG_3857):
            return b
    for b in bs:
        return b
