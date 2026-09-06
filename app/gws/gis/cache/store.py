"""Filesystem tile store, MapProxy 'mp' directory layout."""

import os

import gws
import gws.lib.osx as osx


class Object(gws.TileStore):
    def __init__(self, cache: gws.LayerCache, extension: str):
        self.cache = cache
        self.baseDir = f'{gws.c.MAP_CACHE_DIR}/{cache.name}'
        self.extension = extension

    def count(self) -> int:
        return _file_count(self.baseDir, self.cache.maxAge)

    def count_for_level(self, z: int) -> int:
        return _file_count(f'{self.baseDir}/{z:02d}', self.cache.maxAge)

    def path(self, mt: gws.MapTile) -> str:
        x, y, z = mt
        s = 10000
        return f'{self.baseDir}/{z:02d}/{x // s:04d}/{x % s:04d}/{y // s:04d}/{y % s:04d}.{self.extension}'

    def has(self, mt: gws.MapTile, max_age: int) -> bool:
        p = self.path(mt)
        age = osx.file_age(p)
        return 0 <= age < max_age

    def read(self, mt: gws.MapTile) -> bytes | None:
        if not self.has(mt, self.cache.maxAge):
            return None
        try:
            with open(self.path(mt), 'rb') as fp:
                return fp.read()
        except OSError:
            return None

    def write(self, mt: gws.MapTile, blob: bytes):
        p = self.path(mt)
        osx.mkdir(os.path.dirname(p))
        tmp = f'{p}.{gws.u.random_string(32)}.tmp'
        with open(tmp, 'wb') as fp:
            fp.write(blob)
        os.replace(tmp, p)

    def drop(self):
        if gws.u.is_dir(self.baseDir):
            osx.rmdir(self.baseDir)


##


def _file_count(path: str, max_age: int) -> int:
    if not gws.u.is_dir(path):
        return 0

    last_time = gws.u.stime() - max_age
    c = 0

    for de in osx.find_entries(path):
        if not de.is_file() or de.name.endswith('.tmp'):
            continue
        s = de.stat()
        if s.st_size == 0 or s.st_mtime < last_time:
            continue
        c += 1

    return c
