"""Miscellaneous GIS-related utilities"""

import re

import gws
import gws.lib.crs
import gws.lib.extent
import gws.lib.gis.source
import gws.lib.gis.zoom
import gws.types as t




def best_axis(
        crs: gws.ICrs,
        protocol: gws.OwsProtocol,
        protocol_version: str,
        inverted_crs: t.Optional[t.List[gws.ICrs]] = None
) -> gws.Axis:
    # inverted_axis_crs_list - list of projection refs which are known
    # to have an inverted axis for this service
    # crs_ref - projection we're going to use with the service

    if inverted_crs and crs in inverted_crs:
        return gws.AXIS_YX

    # @TODO some logic to guess the axis, based on crs, service protocol and version
    # see https://docs.geoserver.org/latest/en/user/services/wfs/basics.html#wfs-basics-axis
    return gws.AXIS_XY


def best_bounds(
        crs: gws.ICrs,
        supported_bounds: t.List[gws.Bounds],
        prefer_projected=True
) -> gws.Bounds:
    for b in supported_bounds:
        if b.crs == crs:
            return b

    return t.cast(gws.Bounds, _best_crs_or_bounds(crs, supported_bounds, is_bounds=True, prefer_projected=prefer_projected))


def best_crs(
        crs: gws.ICrs,
        supported_crs: t.List[gws.ICrs],
        prefer_projected=True
) -> gws.ICrs:
    if crs in supported_crs:
        return crs

    return t.cast(gws.ICrs, _best_crs_or_bounds(crs, supported_crs, is_bounds=False, prefer_projected=prefer_projected))


def _best_crs_or_bounds(want_crs, supported, is_bounds: bool, prefer_projected: bool):
    # @TODO find a projection with less errors

    # a crs with the same srid?

    for s in supported:
        crs = s.crs if is_bounds else s
        if crs.same_as(want_crs):
            return s

    # webmercator supported?

    for s in supported:
        crs = s.crs if is_bounds else s
        if prefer_projected and crs.srid == gws.lib.crs.c3857:
            gws.log.debug(f'best_crs: using {crs.srid!r} for {want_crs.srid!r}')
            return s

    # first projected crs?

    for s in supported:
        crs = s.crs if is_bounds else s
        if prefer_projected == crs.is_projected:
            gws.log.debug(f'best_crs: using {crs.srid!r} for {want_crs.srid!r}')
            return s

    # return the first one

    for s in supported:
        crs = s.crs if is_bounds else s
        gws.log.debug(f'best_crs: using {crs.srid!r} for {want_crs.srid!r}')
        return s
