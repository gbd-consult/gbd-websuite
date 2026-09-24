"""Cache seeding."""

import threading
from typing import cast

import gws
import gws.lib.grid
import gws.lib.osx

from . import core

DEFAULT_BLOCK_SIZE = 8
PROGRESS_INTERVAL = 5


def seed(root: gws.Root, opts: core.SeedOptions) -> core.SeedResult:
    defaults = core.SeedOptions(filter=None, levels=[], maxTime=600, concurrency=1)
    opts = cast(core.SeedOptions, gws.u.merge(defaults, opts))
    try:
        with gws.u.server_lock('seed', 0):
            return _run(root, opts)
    except gws.LockBusyError:
        gws.log.info('seed: already running')
        return core.SeedResult(entries=[], seedTime=0, seedStatus='locked')


##


def _run(root: gws.Root, opts: core.SeedOptions) -> core.SeedResult:
    st = core.status(root)
    if opts.filter:
        st = core.apply_filter(st, opts.filter)
    if opts.maxAge is not None:
        for e in st.entries:
            e.grabber.store.maxAge = min(opts.maxAge, e.grabber.cache.maxAge)
    core.add_counts_and_sizes(st)
    core.restrict(st, opts.filter.bbox if opts.filter else None, opts.levels)
    res = core.SeedResult(entries=st.entries, seedTime=0, seedStatus='')
    ts = gws.u.stime()

    queue = _BlockQueue(res.entries)

    deadline = gws.u.stime() + opts.maxTime
    threads = [threading.Thread(target=_worker, args=(queue, deadline), daemon=True) for _ in range(opts.concurrency)]

    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(0.5)
    except KeyboardInterrupt:
        gws.log.info('seed: interrupted')
        queue.stop('interrupted')

    queue.report()

    res.seedTime = gws.u.stime() - ts
    res.seedStatus = queue.seedStatus

    return res


class _BlockGenerator:
    """Blocks of one entry, level by level."""

    def __init__(self, entry: core.Entry, levels: list[core.Level]):
        self.entry = entry
        self.levels = {lv.z: lv for lv in levels}
        self.size = getattr(self.entry.grabber, 'requestTiles', 0) or DEFAULT_BLOCK_SIZE
        self.blocks = self.iter_blocks()
        self.startTime: dict[int, float] = {}

    def iter_blocks(self):
        for z in sorted(self.levels):
            x0, y0, x1, y1, _ = self.levels[z].gridRange
            n = self.size
            for by in range((y0 // n) * n, y1 + 1, n):
                for bx in range((x0 // n) * n, x1 + 1, n):
                    yield max(bx, x0), max(by, y0), min(bx + n - 1, x1), min(by + n - 1, y1), z


class _BlockQueue:
    """Yields blocks to seed, round-robin over entries, so that no single source gets all threads."""

    def __init__(self, entries: list[core.Entry]):
        self.lock = threading.Lock()
        self.entries = entries
        self.generators: list[_BlockGenerator] = []
        self.seedStatus = ''
        self.stopped = False
        self.lastReport = gws.u.stime()

        for e in entries:
            if e.levels:
                self.generators.append(_BlockGenerator(e, e.levels))

    def next_block(self) -> tuple[_BlockGenerator, gws.MapTileRange] | None:
        with self.lock:
            while self.generators:
                bg = self.generators.pop(0)
                block = next(bg.blocks, None)
                if block is None:
                    continue
                z = block[4]
                if z not in bg.startTime:
                    bg.startTime[z] = gws.u.stime()
                    bg.levels[z].cachedTiles = 0
                self.generators.append(bg)
                return bg, block

    def block_complete(self, bg: _BlockGenerator, block: gws.MapTileRange, present: int, fetched: int, failed: int):
        with self.lock:
            z = block[4]
            lv = bg.levels[z]
            lv.cachedTiles += present
            lv.fetchedTiles += fetched
            lv.failedTiles += failed
            lv.seedTime = gws.u.stime() - bg.startTime[z]
            if gws.u.stime() - self.lastReport >= PROGRESS_INTERVAL:
                self.report()

    def stop(self, status: str):
        with self.lock:
            self.seedStatus = status
            self.stopped = True
            for bg in self.generators:
                bg.entry.seedStatus = status

    def report(self):
        self.lastReport = gws.u.stime()
        percents = {e.name: core.percentage_by_level(e) for e in self.entries}
        for name, ps in sorted(percents.items()):
            gws.log.info(f'seed {name}: %% {" ".join(f"{z}:{p}" for z, p in enumerate(ps))}')

def _worker(queue: _BlockQueue, deadline: float):
    while not queue.stopped:
        if gws.u.stime() >= deadline:
            gws.log.info('seed: time limit reached')
            queue.stop('timeout')
            return

        p = queue.next_block()
        if not p:
            return
        bg, block = p

        try:
            present, fetched, failed = _seed_block(queue, bg, block)
        except Exception as exc:
            gws.log.error(f'seed {bg.entry.name}: block {block} error: {exc!r}')
            present, fetched, failed = 0, 0, 0

        if not queue.stopped:
            queue.block_complete(bg, block, present, fetched, failed)


def _seed_block(queue: _BlockQueue, bg: _BlockGenerator, block: gws.MapTileRange) -> tuple[int, int, int]:
    present = 0
    missing = []

    for mt in gws.lib.grid.enum_tiles(block):
        if bg.entry.grabber.store.has(mt, bg.entry.grabber.store.maxAge):
            present += 1
        else:
            missing.append(mt)

    if not missing:
        return present, 0, 0

    try:
        for mt in missing:
            if queue.stopped:
                return present, 0, 0
            bg.entry.grabber.get_tile_as_bytes(mt)
        return present, len(missing), 0
    except Exception as exc:
        gws.log.warning(f'seed {bg.entry.name}: block {block} failed: {exc!r}')
        return present, 0, len(missing)
