"""Filesystem tile store, MapProxy 'mp' directory layout."""

import os

import gws
import gws.lib.osx as osx


class Object(gws.TileStore):
    def __init__(self, base_dir: str, max_age: int, extension: str):
        self.baseDir = base_dir
        self.maxAge = max_age
        self.extension = extension

    def stats(self) -> gws.TileStoreStats:
        return _stats(self.baseDir, self.maxAge, None)

    def stats_for_level(self, z: int) -> gws.TileStoreStats:
        return _stats(f'{self.baseDir}/{z:02d}', self.maxAge, z)

    def path(self, mt: gws.MapTile) -> str:
        x, y, z = mt
        s = 10000
        return f'{self.baseDir}/{z:02d}/{x // s:04d}/{x % s:04d}/{y // s:04d}/{y % s:04d}.{self.extension}'

    def has(self, mt: gws.MapTile, max_age: int) -> bool:
        p = self.path(mt)
        age = osx.file_age(p)
        return 0 <= age < max_age

    def read(self, mt: gws.MapTile) -> bytes | None:
        if not self.has(mt, self.maxAge):
            return None
        try:
            with open(self.path(mt), 'rb') as fp:
                return fp.read()
        except OSError:
            return None

    def write(self, mt: gws.MapTile, blob: bytes):
        p = self.path(mt)
        try:
            osx.mkdir(os.path.dirname(p))
            gws.u.write_file_b(p, blob)
        except OSError as exc:
            gws.log.warning(f'tile store: write failed {p!r}: {exc}')

    def drop(self):
        if gws.u.is_dir(self.baseDir):
            osx.rmdir(self.baseDir)

    def drop_level(self, z: int):
        path = f'{self.baseDir}/{z:02d}'
        if gws.u.is_dir(path):
            osx.rmdir(path)

    def drop_range(self, mtr: gws.MapTileRange):
        x0, y0, x1, y1, z = mtr
        s = 10000
        level_dir = f'{self.baseDir}/{z:02d}'

        for xh, xh_path in _numbered_entries(level_dir, x0 // s, x1 // s):
            for _, xl_path in _numbered_entries(xh_path, x0 - xh * s, x1 - xh * s):
                for yh, yh_path in _numbered_entries(xl_path, y0 // s, y1 // s):
                    for _, yl_path in _numbered_entries(yh_path, y0 - yh * s, y1 - yh * s):
                        osx.unlink(yl_path)
                    _rmdir_if_empty(yh_path)
                _rmdir_if_empty(xl_path)
            _rmdir_if_empty(xh_path)
        _rmdir_if_empty(level_dir)


##


def _numbered_entries(path: str, lo: int, hi: int) -> list[tuple[int, str]]:
    if not gws.u.is_dir(path):
        return []
    res = []
    for de in os.scandir(path):
        stem = de.name.split('.')[0]
        if stem.isdigit() and lo <= int(stem) <= hi:
            res.append((int(stem), de.path))
    return res


def _rmdir_if_empty(path: str):
    try:
        os.rmdir(path)
    except OSError:
        pass


def _stats(path: str, max_age: int, z: int | None) -> gws.TileStoreStats:
    st = gws.TileStoreStats(count=0, size=0, range=None)
    if not gws.u.is_dir(path):
        return st

    last_time = gws.u.stime() - max_age
    rng = None

    for de in osx.find_entries(path):
        if not de.is_file() or de.name.endswith('.tmp'):
            continue
        s = de.stat()
        if s.st_size == 0 or s.st_mtime < last_time:
            continue
        st.count += 1
        st.size += s.st_size
        if z is None:
            continue
        a, b, c, d = de.path[len(path) + 1 :].rsplit('.', 1)[0].split('/')
        x = int(a) * 10000 + int(b)
        y = int(c) * 10000 + int(d)
        if rng is None:
            rng = [x, y, x, y]
        else:
            rng = [min(rng[0], x), min(rng[1], y), max(rng[2], x), max(rng[3], y)]

    if rng:
        st.range = rng[0], rng[1], rng[2], rng[3], z
    return st
