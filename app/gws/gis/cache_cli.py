from argh import arg
import time
import gws
import gws.config
import gws.config.loader
import gws.lib.misc
import gws.lib.units as units
import gws.lib.clihelpers as clihelpers
import gws.gis.cache

COMMAND = 'cache'


@arg('--layers', help='comma separated list of layer IDs')
def status(layers=None):
    """Display the cache status"""

    root = gws.config.loader.load()

    if layers:
        layers = _as_list(layers)

    st = gws.gis.cache.status(root, layers)

    if not st:
        print('no cached layers found')
        return

    for la_uid, info in sorted(st.items()):
        print()
        print(f"CACHE  : {info['cache_uid']}")
        print(f"CONFIG : {repr(info['config'])}")
        print(f"LAYER  :")
        for uid in info['layer_uids']:
            print(f"    {uid}")
        print()

        data = []

        for r in info['counts']:
            data.append({
                'zoom': r['z'],
                'scale': round(units.res2scale(r['res'])),
                'grid': str(r['maxx']) + 'x' + str(r['maxy']),
                'total': r['maxx'] * r['maxy'],
                'cached': r['num_files'],
                '%%': int(100 * (r['num_files'] / (r['maxx'] * r['maxy']))),
            })
        print(clihelpers.text_table(data, ['zoom', 'scale', 'grid', 'total', 'cached', '%%']))

    print()
    print()

    u = gws.gis.cache.dangling_dirs(root)
    if u:
        print(f'{len(u)} DANGLING CACHES:')
        print()
        print('\n'.join(u))

    print()


def clean():
    """Clean up the cache."""

    root = gws.config.loader.load()
    gws.gis.cache.clean(root)


@arg('--layers', help='comma separated list of layer IDs')
def drop(layers=None):
    """Drop caches for specific or all layers."""

    root = gws.config.loader.load()
    if layers:
        layers = _as_list(layers)
    gws.gis.cache.drop(root, layers)


_SEED_LOCKFILE = gws.CONFIG_DIR + '/mapproxy.seed.lock'


@arg('--layers', help='comma separated list of layer IDs')
@arg('--levels', help='zoom levels to build the cache for')
def seed(layers=None, levels=None):
    """Start the cache seeding process"""

    root = gws.config.loader.load()

    if layers:
        layers = _as_list(layers)

    st = gws.gis.cache.status(root, layers)

    if not st:
        print('no cached layers found')
        return

    if levels:
        levels = [int(x) for x in _as_list(levels)]

    with gws.lib.misc.lock(_SEED_LOCKFILE) as ok:
        if not ok:
            print('seed already running')
            return

        max_time = root.var('seeding.maxTime')
        concurrency = root.var('seeding.concurrency')
        ts = time.time()

        print(f'\nSTART SEEDING (maxTime={max_time} concurrency={concurrency}), ^C ANYTIME TO CANCEL...\n')
        done = gws.gis.cache.seed(root, layers, max_time, concurrency, levels)
        print('=' * 40)
        print('TIME: %.1f sec' % (time.time() - ts))
        print(f'SEEDING COMPLETE' if done else 'SEEDING INCOMPLETE, PLEASE TRY AGAIN')


def _as_list(s):
    ls = []
    for p in s.split(','):
        p = p.strip()
        if p:
            ls.append(p)
    return ls
