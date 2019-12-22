from argh import arg
import gws
import gws.config
import gws.config.loader
import gws.tools.clihelpers
import gws.tools.json2
import gws.gis.mpx.config

COMMAND = 'config'


@arg('--path', help='configuration file path')
def test(path=None):
    """Run a configuration file test"""

    gws.config.loader.parse_and_activate(path)
    if gws.config.root().var('server.mapproxy.enabled'):
        gws.gis.mpx.config.create_and_save('/tmp/mapproxy-check')
    gws.log.info('CONFIGURATION OK')


@arg('--path', help='configuration file path')
def prepare(path=None):
    """Parse and prepare a config"""

    gws.config.loader.parse_and_activate(path)
    gws.config.loader.store()


@arg('--path', help='configuration file path')
@arg('--out', help='path to write the dump to')
def dump(path=None, out=None):
    """Dump the configuarion tree"""

    if path:
        gws.config.loader.parse_and_activate(path)
    else:
        gws.config.loader.load()

    r = gws.tools.json2.to_tagged_dict(gws.config.root())
    r = gws.tools.json2.to_string(r, pretty=True)

    if not out:
        print(r)
        return

    with open(out, 'wt') as fp:
        fp.write(r)
