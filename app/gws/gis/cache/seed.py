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
    """Fill the selected caches with missing tiles.

    Tiles are requested from the grabbers in blocks by ``opts.concurrency`` worker threads,
    round-robin over the caches, until all blocks are done, the time limit is reached
    or the run is interrupted. Only one seeding run can be active on the server;
    if another one is running, nothing is done and the status is ``locked``.

    Args:
        root: Configuration root.
        opts: Seeding options. Missing values default to 600 seconds and one thread.

    Returns:
        The seeding result with per-level statistics.
    """

    defaults = core.SeedOptions(filter=None, maxTime=600, concurrency=1)
    opts = cast(core.SeedOptions, gws.u.merge(defaults, opts))
    try:
        with gws.u.server_lock('seed', 0):
            return _run(root, opts)
    except gws.LockBusyError:
        gws.log.info('seed: already running')
        return core.SeedResult(caches=[], seedTime=0, seedStatus='locked')


##


def _run(root: gws.Root, opts: core.SeedOptions) -> core.SeedResult:
    inv = core.inventory(root)
    if opts.filter:
        core.apply_filter(inv, opts.filter)
    if opts.maxAge is not None:
        for c in inv.caches:
            c.grabber.store.maxAge = min(opts.maxAge, c.grabber.cache.maxAge)
    core.add_stats(inv)
    res = core.SeedResult(caches=inv.caches, seedTime=0, seedStatus='')
    ts = gws.u.stime()

    queue = _BlockQueue(res.caches)

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
    """Generates the blocks of one cache, level by level."""

    def __init__(self, cache: core.Cache, levels: list[core.Level]):
        self.cache = cache
        self.levels = {lv.z: lv for lv in levels}
        self.size = getattr(self.cache.grabber, 'requestTiles', 0) or DEFAULT_BLOCK_SIZE
        self.blocks = self.iter_blocks()
        self.startTime: dict[int, float] = {}

    def iter_blocks(self):
        """Yield tile ranges of block size, aligned to multiples of the block size."""
        for z in sorted(self.levels):
            x0, y0, x1, y1, _ = self.levels[z].gridRange
            n = self.size
            for by in range((y0 // n) * n, y1 + 1, n):
                for bx in range((x0 // n) * n, x1 + 1, n):
                    yield max(bx, x0), max(by, y0), min(bx + n - 1, x1), min(by + n - 1, y1), z


class _BlockQueue:
    """Yields blocks to seed, round-robin over caches, so that no single source gets all threads."""

    def __init__(self, caches: list[core.Cache]):
        self.lock = threading.Lock()
        self.caches = caches
        self.generators: list[_BlockGenerator] = []
        self.seedStatus = ''
        self.stopped = False
        self.lastReport = gws.u.stime()

        for c in caches:
            if c.levels:
                self.generators.append(_BlockGenerator(c, c.levels))

    def next_block(self) -> tuple[_BlockGenerator, gws.MapTileRange] | None:
        """Return the next block and its generator, or ``None`` if all blocks are done."""
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
        """Add the counts of a completed block to its level and report progress periodically."""
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
        """Stop seeding and set the status of the run and of the unfinished caches."""
        with self.lock:
            self.seedStatus = status
            self.stopped = True
            for bg in self.generators:
                bg.cache.seedStatus = status

    def report(self):
        """Log the cached percentages of all caches."""
        self.lastReport = gws.u.stime()
        percents = {c.name: core.percentage_by_level(c) for c in self.caches}
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
            gws.log.error(f'seed {bg.cache.name}: block {block} error: {exc!r}')
            present, fetched, failed = 0, 0, 0

        if not queue.stopped:
            queue.block_complete(bg, block, present, fetched, failed)


def _seed_block(queue: _BlockQueue, bg: _BlockGenerator, block: gws.MapTileRange) -> tuple[int, int, int]:
    present = 0
    missing = []

    for mt in gws.lib.grid.enum_tiles(block):
        if bg.cache.grabber.store.has(mt):
            present += 1
        else:
            missing.append(mt)

    if not missing:
        return present, 0, 0

    try:
        for mt in missing:
            if queue.stopped:
                return present, 0, 0
            bg.cache.grabber.get_tile_as_bytes(mt)
        return present, len(missing), 0
    except Exception as exc:
        gws.log.warning(f'seed {bg.cache.name}: block {block} failed: {exc!r}')
        return present, 0, len(missing)
