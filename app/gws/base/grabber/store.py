"""Filesystem tile store, MapProxy 'mp' directory layout."""

import os
import threading

import gws
import gws.lib.osx


class Object:
    def __init__(self, base_dir: str, extension: str):
        self.baseDir = base_dir
        self.extension = extension

    def path(self, mt: gws.MapTile) -> str:
        x, y, z = mt
        return '{}/{:02d}/{:04d}/{:04d}/{:04d}/{:04d}.{}'.format(
            self.baseDir, z, x // 10000, x % 10000, y // 10000, y % 10000, self.extension)

    def read(self, mt: gws.MapTile, max_age: int) -> bytes | None:
        p = self.path(mt)
        age = gws.lib.osx.file_age(p)
        if 0 <= age < max_age:
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
