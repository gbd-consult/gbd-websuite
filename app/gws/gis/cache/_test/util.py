"""Test helpers for the cache package."""

import os

import gws
import gws.base.grabber.box
import gws.base.grabber.core
import gws.lib.crs
import gws.lib.extent
import gws.lib.image
import gws.lib.mime
import gws.lib.osx

RED = (255, 0, 0, 255)
PNG = gws.ImageFormat(name='png', mimeTypes=[gws.lib.mime.PNG], options={})


class Grabber(gws.base.grabber.box.Object):
    def __init__(self, opts):
        super().__init__(opts)
        self.sourceCrs = self.targetCrs
        self.fetches = []
        self.error = None

    def fetch_box_as_image(self, bounds, width, height, params=None):
        self.fetches.append((bounds.extent, width, height))
        if self.error:
            raise self.error
        return gws.lib.image.from_size((width, height), color=RED)


class Root:
    def __init__(self, layers):
        self.layers = layers

    def find_all(self, cls):
        return self.layers


def system_dirs():
    gws.u.ensure_dir(gws.c.LOCKS_DIR)
    gws.u.ensure_dir(gws.c.EPHEMERAL_DIR)
    gws.u.ensure_dir(gws.c.MAP_CACHE_DIR)


def cache_name():
    return 'cache_' + gws.u.random_string(8)


def grabber(name=None, max_age=3600, max_level=3, srid=3857) -> Grabber:
    crs = gws.lib.crs.get(srid)
    cache = gws.MapCache(
        name=name or cache_name(),
        maxAge=max_age,
        maxLevel=max_level,
        requestBuffer=0,
        requestTiles=4,
    )
    opts = gws.base.grabber.core.Options(
        crs=crs,
        cache=cache,
        extent=gws.lib.extent.transform_from_wgs(crs.wgsMaxExtent, crs),
        imageFormat=PNG,
    )
    return Grabber(opts)


def layer(uid, grabbers, title='', ext_type='type_1'):
    return gws.Data(uid=uid, title=title, extType=ext_type, grabbers={gr.targetCrs.srid: gr for gr in grabbers})


def make_old(path, age):
    t = gws.u.stime() - age
    for p in [path] if os.path.isfile(path) else gws.lib.osx.find_files(path):
        os.utime(p, (t, t))
