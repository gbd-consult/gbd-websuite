"""Filesystem tile store, MapProxy 'mp' directory layout."""

import os

import gws
import gws.lib.osx as osx


class Object(gws.TileStore):
    """Filesystem tile store."""

    def __init__(self, base_dir: str, max_age: int, extension: str):
        """Create a store.

        Args:
            base_dir: Base directory.
            max_age: Max. age of stored tiles in seconds.
            extension: File extension of tile files, without a dot.
        """
        self.baseDir = base_dir
        self.maxAge = max_age
        self.extension = extension

    def stats(self) -> gws.TileStoreStats:
        return self._stats()

    def stats_for_level(self, z: int) -> gws.TileStoreStats:
        return self._stats(z)

    def path(self, mt: gws.MapTile) -> str:
        x, y, z = mt
        s = 10000
        return f'{self._level_dir(z)}/{(x // s):04d}/{(x % s):04d}/{(y // s):04d}/{(y % s):04d}.{self.extension}'

    def has(self, mt: gws.MapTile) -> bool:
        p = self.path(mt)
        age = osx.file_age(p)
        return 0 <= age < self.maxAge

    def read(self, mt: gws.MapTile) -> bytes | None:
        if not self.has(mt):
            return None
        try:
            with open(self.path(mt), 'rb') as fp:
                return fp.read()
        except OSError:
            return None

    def write(self, mt: gws.MapTile, blob: bytes):
        p = self.path(mt)
        try:
            gws.u.ensure_dir(os.path.dirname(p))
            gws.u.write_file_b(p, blob)
        except OSError as exc:
            gws.log.warning(f'tile store: write failed {p!r}: {exc}')

    def drop(self):
        osx.rmdir(self.baseDir)

    def drop_level(self, z: int):
        osx.rmdir(self._level_dir(z))

    def drop_range(self, mtr: gws.MapTileRange):
        x0, y0, x1, y1, z = mtr
        dir = self._level_dir(z)
        if not gws.u.is_dir(dir):
            return

        for de in osx.find_entries(dir):
            if not de.is_file() or de.name.endswith('.tmp'):
                continue
            x, y = self._tile_xy(de.path)
            if x0 <= x <= x1 and y0 <= y <= y1:
                osx.unlink(de.path)

    ##

    def _level_dir(self, z: int) -> str:
        return f'{self.baseDir}/{z:02d}'

    def _tile_xy(self, path: str) -> tuple[int, int]:
        a, b, c, d = path.rsplit('.', 1)[0].split('/')[-4:]
        return int(a) * 10000 + int(b), int(c) * 10000 + int(d)

    def _stats(self, z: int | None = None) -> gws.TileStoreStats:
        stats = gws.TileStoreStats(count=0, size=0, range=None)
        dir = self._level_dir(z) if z is not None else self.baseDir
        if not gws.u.is_dir(dir):
            return stats

        last_time = gws.u.stime() - self.maxAge

        for de in osx.find_entries(dir):
            if not de.is_file() or de.name.endswith('.tmp'):
                continue
            s = de.stat()
            if s.st_size == 0 or s.st_mtime < last_time:
                continue
            stats.count += 1
            stats.size += s.st_size
            if z is None:
                continue

            x, y = self._tile_xy(de.path)
            if stats.range is None:
                stats.range = (x, y, x, y, z)
            else:
                stats.range = (
                    min(stats.range[0], x),
                    min(stats.range[1], y),
                    max(stats.range[2], x),
                    max(stats.range[3], y),
                    z,
                )

        return stats
