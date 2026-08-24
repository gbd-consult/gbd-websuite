"""Tests for the qfieldcloud base map rendering. Requires the qgis server."""

from typing import cast

import os
import time

import gws
import gws.lib.gdalx
import gws.test.util as u

from gws.plugin.qfieldcloud import action as action_mod, packager
from gws.plugin.qfieldcloud._test import util as tu

CONFIG = """
    projects+ {
        uid "PROJECT_1"
        access "allow all"
        actions+ {
            type "qfieldcloud"
            uid "ACTION_1"
            access "allow all"
            projects+ {
                uid "QFC_1"
                title "QField Test"
                access "allow all"
                provider.path {QGS_PATH}
                mapCacheLifeTime {CACHE_TIME}
            }
        }
    }
"""


def _aoi(x, y, size):
    return (
        f'Polygon (({x} {y}, {x + size} {y}, {x + size} {y + size}, {x} {y + size}, {x} {y}))'
    )


def _root(name, patch=None, cache_time=0):
    # NB `mapCacheLifeTime` is a `gws.Duration`, but `u.gws_root` does not run the
    # spec reader, so it has to be given as a plain number of seconds here
    return u.gws_root(
        CONFIG,
        QGS_PATH=repr(tu.qgs_path(name, patch)),
        CACHE_TIME=cache_time,
    )


def _render(root, cache_dir):
    act = cast(action_mod.Object, root.get('ACTION_1'))
    qp = act.qfcProjects[0]

    pa = packager.Args(
        uid='TEST',
        qfcProject=qp,
        caps=act.get_caps(qp),
        project=root.app.project('PROJECT_1'),
        user=root.app.authMgr.systemUser,
        packageDir=str(cache_dir),
        mapCacheDir=gws.u.ensure_dir(str(cache_dir) + '/cache'),
        withBaseMap=True,
        withData=False,
        withMedia=False,
        withQgis=False,
    )

    po = packager.Object()
    po.create_package(root, pa)
    return po


@u.fixture(scope='module')
def root():
    tu.create_tables()
    u.pg.insert('qfc.district', [
        {'id': 1, 'name': 'one', 'geom': tu.polygon(744000, 6644000, 5000)},
    ])
    yield _root('basemap')


##


def test_base_map_is_rendered(root: gws.Root, tmp_path):
    po = _render(root, tmp_path)

    name = gws.u.to_uid('basemap_L9') + '.gpkg'
    path = po.pathMap[name]

    assert os.path.basename(path) == f'12_{name}'
    assert os.path.isfile(path)

    with gws.lib.gdalx.open_raster(path) as ds:
        assert ds.size()[0] > 0


def test_base_map_is_not_cached_by_default(root: gws.Root, tmp_path):
    po = _render(root, tmp_path)
    path = list(po.pathMap.values())[0]

    t = time.time() - 60
    os.utime(path, (t, t))
    _render(root, tmp_path)

    assert os.path.getmtime(path) > t + 1


def test_base_map_is_cached(tmp_path):
    root = _root('basemap_cached', cache_time=3600)

    po = _render(root, tmp_path)
    path = list(po.pathMap.values())[0]

    t = time.time() - 60
    os.utime(path, (t, t))
    po = _render(root, tmp_path)

    u.check.close(os.path.getmtime(path), t, abs_tol=1)
    assert list(po.pathMap.values())[0] == path


def test_stale_cache_is_rerendered(tmp_path):
    root = _root('basemap_stale', cache_time=60)

    po = _render(root, tmp_path)
    path = list(po.pathMap.values())[0]

    t = time.time() - 3600
    os.utime(path, (t, t))
    _render(root, tmp_path)

    assert os.path.getmtime(path) > t + 1


def test_zoom_level_is_clamped_at_the_bottom(tmp_path):
    def patch(root_el):
        tu.set_project_prop(root_el, 'baseMapTilesMaxZoomLevel', '1', 'int')
        tu.set_project_prop(root_el, 'baseMapTilesMinZoomLevel', '1', 'int')
        tu.set_project_prop(root_el, 'areaOfInterest', _aoi(744000, 6644000, 200000))

    po = _render(_root('basemap_zoom_low', patch), tmp_path)
    path = list(po.pathMap.values())[0]

    assert os.path.basename(path).startswith('3_')


def test_zoom_level_is_clamped_at_the_top(tmp_path):
    def patch(root_el):
        tu.set_project_prop(root_el, 'baseMapTilesMaxZoomLevel', '30', 'int')
        tu.set_project_prop(root_el, 'baseMapTilesMinZoomLevel', '30', 'int')
        tu.set_project_prop(root_el, 'areaOfInterest', _aoi(744000, 6644000, 100))

    po = _render(_root('basemap_zoom_high', patch), tmp_path)
    path = list(po.pathMap.values())[0]

    assert os.path.basename(path).startswith('20_')
