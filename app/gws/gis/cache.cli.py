from argh import arg
import time
import gws
import gws.config
import gws.config.loader
import gws.tools.misc as misc
import gws.gis.cache
import gws.types as t

COMMAND = 'cache'


@arg('--layers', help='comma separated list of layer IDs')
def updateweb(layers=None):
    """Update the front (web) cache from the back cache"""

    gws.config.loader.load()

    if layers:
        layers = _as_list(layers)

    fs = gws.gis.cache.updateweb(layers)


@arg('--layers', help='comma separated list of layer IDs')
def status(layers=None):
    """Display the cache status"""

    gws.config.loader.load()

    if layers:
        layers = _as_list(layers)

    st = gws.gis.cache.status(layers)

    for la_uid, rs in sorted(st.items()):
        for r in rs:
            r['la'] = la_uid
            r['total'] = r['maxx'] * r['maxy']
            r['percent'] = int(100 * (r['num_files'] / r['total']))
            gws.log.info('layer={la} z={z} ({res}) {maxx}x{maxy}={total:,} cached={num_files} ({percent}%)'.format(**r))
        gws.log.info('-' * 40)


@arg('--layers', help='comma separated list of layer IDs')
def drop(layers=None):
    """Drop the caches"""

    gws.config.loader.load()
    if layers:
        layers = _as_list(layers)
    gws.gis.cache.drop(layers)


_lockfile = gws.CONFIG_DIR + '/mapproxy.seed.lock'


@arg('--layers', help='comma separated list of layer IDs')
@arg('--level', help='zoom level to build the cache for')
def seed(layers=None, level=None):
    """Start the cache seeding process"""

    gws.config.loader.load()

    with misc.lock(_lockfile) as ok:
        if not ok:
            gws.log.info('seed already running')
            return

        if layers:
            layers = _as_list(layers)

        max_time = gws.config.var('seeding.maxTime')
        concurrency = gws.config.var('seeding.concurrency')
        ts = time.time()

        gws.log.info(f'START SEEDING (maxTime={max_time}), ^C ANYTIME TO CANCEL...')
        ok = gws.gis.cache.seed(layers, max_time, concurrency, level)
        gws.log.info('=' * 40)
        gws.log.info('TIME: %.1f sec' % (time.time() - ts))
        gws.log.info(f'SEEDING COMPLETE' if ok else 'SEEDING INCOMPLETE, PLEASE TRY AGAIN')


def _as_list(s):
    ls = []
    for p in s.split(','):
        p = p.strip()
        if p:
            ls.append(p)
    return ls
