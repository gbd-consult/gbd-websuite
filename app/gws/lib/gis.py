"""Miscellaneous GIS-related utilities"""

import re

import gws
import gws.lib.extent
import gws.lib.metadata
import gws.lib.proj
import gws.lib.proj
import gws.types as t


def make_bbox(x, y, crs, resolution, pixel_width, pixel_height):
    """Given a point (in crs units), make a widthXheight pixel bbox around it."""

    # @TODO

    # is_geographic = gws.lib.proj.is_geographic(crs)
    #
    # if is_geographic:
    #     x, y = gws.lib.proj.transform_xy(x, y, crs, 'EPSG:3857')

    bbox = [
        x - (pixel_width * resolution) / 2,
        y - (pixel_height * resolution) / 2,
        x + (pixel_width * resolution) / 2,
        y + (pixel_height * resolution) / 2,
    ]

    # if is_geographic:
    #     bbox = gws.lib.proj.transform_extent(bbox, 'EPSG:3857', crs)

    return bbox


def invert_bbox(bbox):
    return [bbox[1], bbox[0], bbox[3], bbox[2]]


def best_axis(crs, inverted_axis_crs_list, protocol: gws.OwsProtocol, protocol_version):
    # inverted_axis_crs_list - list of EPSG crs'es which are known to have an inverted axis for this service
    # crs - crs we're going to use with the service

    proj = gws.lib.proj.as_proj(crs)
    if inverted_axis_crs_list and proj.epsg in inverted_axis_crs_list:
        return 'yx'

    # @TODO some logic to guess the axis, based on crs, service protocol and version
    # see https://docs.geoserver.org/latest/en/user/services/wfs/basics.html#wfs-basics-axis
    return 'xy'


def best_crs(target_crs, supported_crs):
    # find the best matching crs for the target crs and the list of supported crs.

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
        p = gws.lib.proj.as_proj(crs)
        if p and not p.is_geographic:
            gws.log.debug(f'best_crs: using {p.epsg!r} for {target_crs!r}')
            return p.epsg

    raise gws.Error(f'no match for crs={target_crs!r} in {supported_crs!r}')


def best_crs_and_shape(request_crs, supported_crs, shape):
    crs = best_crs(request_crs, supported_crs)
    return crs, shape.transformed_to(crs)


class SourceStyle(gws.Data):
    is_default: bool
    legend_url: gws.Url
    metadata: gws.lib.metadata.Record
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

    metadata: gws.lib.metadata.Record
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