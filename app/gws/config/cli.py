from argh import arg

import gws
import gws.types as t
import gws.config
import gws.config.loader
import gws.lib.mpx.config
import gws.lib.clihelpers
import gws.lib.json2

COMMAND = 'config'


@arg('--path', help='configuration file path')
def test(path=None):
    """Run a configuration file test"""

    root = gws.config.loader.parse_and_activate(path)
    if root.application.var('server.mapproxy.enabled'):
        gws.lib.mpx.config.create_and_save(root, '/tmp/mapproxy-check')
    gws.log.info('CONFIGURATION OK')


@arg('--path', help='configuration file path')
def prepare(path=None):
    """Parse and prepare a config"""

    root = gws.config.loader.parse_and_activate(path)
    gws.config.loader.store(root)


@arg('--path', help='configuration file path')
@arg('--out', help='path to write the dump to')
def dump(path=None, out=None):
    """Dump the configuarion tree"""

    if path:
        root = gws.config.loader.parse_and_activate(path)
    else:
        root = gws.config.loader.load()

    r = gws.lib.json2.to_tagged_string(root, pretty=True, ascii=False)

    if not out:
        print(r)
        return

    with open(out, 'wt') as fp:
        fp.write(r)

@arg('--path', help='configuration file path')
def dumpmeta(path=None, out=None):
    """Dump object metadata from the configuarion tree"""

    if path:
        root = gws.config.loader.parse_and_activate(path)
    else:
        root = gws.config.loader.load()

    ls = []

    for obj in root.find_all():
        meta = gws.get(obj, 'meta')
        if meta:
            tag = '$%s.%s:%s' % (
                getattr(obj, '__module__', ''),
                obj.__class__.__name__,
                obj.uid)
            ls.append({
                '$': tag,
                'meta': meta
            })

    r = gws.lib.json2.to_pretty_string(ls, ascii=False)
    print(r)
