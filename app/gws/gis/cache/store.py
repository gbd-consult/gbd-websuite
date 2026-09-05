"""Filesystem tile store, MapProxy 'mp' directory layout."""

import os
import threading

import gws
import gws.lib.osx


class Object:
    def __init__(self, cache: gws.LayerCache, extension: str):
        self.cache = cache
        self.baseDir = f'{gws.c.MAP_CACHE_DIR}/{cache.name}'
        self.extension = extension

    def path(self, mt: gws.MapTile) -> str:
        x, y, z = mt
        s = 10000
        return f'{self.baseDir}/{z:02d}/{x // s:04d}/{x % s:04d}/{y // s:04d}/{y % s:04d}.{self.extension}'

    def read(self, mt: gws.MapTile) -> bytes | None:
        p = self.path(mt)
        age = gws.lib.osx.file_age(p)
        if 0 <= age < self.cache.maxAge:
            try:
                with open(p, 'rb') as fp:
                    return fp.read()
            except OSError:
                return None

    def write(self, mt: gws.MapTile, blob: bytes):
        p = self.path(mt)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f'{p}.{os.getpid()}.{threading.get_ident()}.tmp'
        with open(tmp, 'wb') as fp:
            fp.write(blob)
        os.replace(tmp, p)

    def drop(self):
        if os.path.isdir(self.baseDir):
            gws.lib.osx.rmdir(self.baseDir)
