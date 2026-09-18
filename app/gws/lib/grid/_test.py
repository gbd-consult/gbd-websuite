"""Tests for the grid module."""

import gws
import gws.lib.crs
import gws.lib.grid as grid
import gws.test.util as u

CRS_3857 = gws.lib.crs.get(3857)
GRID_3857 = grid.for_crs(CRS_3857)
TMS = grid.matrix_set_for_grid(GRID_3857, 20)
M = TMS.matrices
HALF = 20037508.342789244


def _tile_extent(mt):
    return grid.extent_for_tile(GRID_3857, mt)


def test_matrix_set_for_grid():
    assert TMS.crs == CRS_3857
    assert [m.identifier for m in M[:3]] == ['0', '1', '2']
    assert abs(M[0].resolution - 156543.03392804097) < 1e-6
    assert abs(M[12].resolution - 156543.03392804097 / 4096) < 1e-9
    assert (M[12].width, M[12].height) == (4096, 4096)
    assert (M[0].x, M[0].y) == (-HALF, HALF)


def test_matrix_range_for_extent():
    assert grid.matrix_range_for_extent(M[0], M[0].extent) == (0, 0, 0, 0, 0)
    assert grid.matrix_range_for_extent(M[1], (0, 0, HALF, HALF)) == (1, 0, 1, 0, 0)
    assert grid.matrix_range_for_extent(M[1], (-1e5, -1e5, 1e5, 1e5)) == (0, 0, 1, 1, 0)
    assert grid.matrix_range_for_extent(M[12], _tile_extent((2124, 1364, 12))) == (2124, 1364, 2124, 1364, 0)
    assert grid.matrix_range_for_extent(M[1], (HALF + 1, 0, HALF + 2, 1)) is None
    assert grid.matrix_range_for_extent(M[1], (0, -HALF - 2, 1, -HALF - 1)) is None
    assert grid.matrix_range_for_extent(M[1], (-HALF - 10, -HALF - 10, HALF + 10, HALF + 10)) == (0, 0, 1, 1, 0)


def test_matrix_extent_for_range():
    assert grid.matrix_extent_for_range(M[1], (1, 0, 1, 0, 0)) == (0, 0, HALF, HALF)
    e = grid.matrix_extent_for_range(M[12], (2124, 1364, 2125, 1365, 0))
    assert e == grid.extent_for_range(GRID_3857, (2124, 1364, 2125, 1365, 12))


def test_matrix_for_resolution_never_upscales():
    r5 = M[5].resolution
    assert grid.matrix_for_resolution(TMS, r5).identifier == '5'
    assert grid.matrix_for_resolution(TMS, r5 * 1.5).identifier == '5'
    assert grid.matrix_for_resolution(TMS, r5 * 0.9).identifier == '6'
    assert grid.matrix_for_resolution(TMS, r5 * 2).identifier == '4'
    assert grid.matrix_for_resolution(TMS, M[20].resolution / 10).identifier == '20'


def test_matrix_for_resolution_tolerates_rounding():
    r5 = M[5].resolution
    assert grid.matrix_for_resolution(TMS, r5 * (1 - 0.5 * grid.RESOLUTION_TOLERANCE)).identifier == '5'
    assert grid.matrix_for_resolution(TMS, r5 * (1 - 2 * grid.RESOLUTION_TOLERANCE)).identifier == '6'


def test_matrix_for_resolution_unordered_matrices():
    tms = gws.TileMatrixSet(identifier='', crs=CRS_3857, matrices=[M[7], M[3], M[5]])
    assert grid.matrix_for_resolution(tms, M[5].resolution * 1.5).identifier == '5'
    assert grid.matrix_for_resolution(tms, M[20].resolution).identifier == '7'


def test_level_for_resolution():
    r5 = grid.resolution_for_level(GRID_3857, 5)
    assert grid.level_for_resolution(GRID_3857, r5) == 5
    assert grid.level_for_resolution(GRID_3857, r5 * 1.5) == 5
    assert grid.level_for_resolution(GRID_3857, r5 * (1 - 0.5 * grid.RESOLUTION_TOLERANCE)) == 5
    assert grid.level_for_resolution(GRID_3857, r5 * (1 - 2 * grid.RESOLUTION_TOLERANCE)) == 6
