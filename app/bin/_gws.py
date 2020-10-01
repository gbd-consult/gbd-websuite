import sys

import argh

import gws
import gws.tools.clihelpers as ch

gws.ensure_dir(gws.MAPPROXY_CACHE_DIR)
gws.ensure_dir(gws.LEGEND_CACHE_DIR)
gws.ensure_dir(gws.NET_CACHE_DIR)
gws.ensure_dir(gws.OBJECT_CACHE_DIR)
gws.ensure_dir(gws.WEB_CACHE_DIR)
gws.ensure_dir(gws.CONFIG_DIR)
gws.ensure_dir(gws.LOG_DIR)
gws.ensure_dir(gws.MISC_DIR)
gws.ensure_dir(gws.PRINT_DIR)
gws.ensure_dir(gws.SERVER_DIR)
gws.ensure_dir(gws.SPOOL_DIR)

COMMANDS = {}

import gws.common.auth.cli
import gws.config.cli
import gws.ext.action.bplan.cli
import gws.ext.action.dprocon.cli
import gws.ext.action.gekos.cli
import gws.ext.action.georisks.cli
import gws.ext.helper.alkis.cli
import gws.gis.cache_cli
import gws.server.cli

COMMANDS['alkis'] = [gws.ext.helper.alkis.cli.check_index, gws.ext.helper.alkis.cli.drop_index, gws.ext.helper.alkis.cli.parse, gws.ext.helper.alkis.cli.setup]
COMMANDS['auth'] = [gws.common.auth.cli.clear, gws.common.auth.cli.passwd, gws.common.auth.cli.sessions, gws.common.auth.cli.test]
COMMANDS['bplan'] = [gws.ext.action.bplan.cli.read, gws.ext.action.bplan.cli.update]
COMMANDS['cache'] = [gws.gis.cache_cli.clean, gws.gis.cache_cli.drop, gws.gis.cache_cli.seed, gws.gis.cache_cli.status]
COMMANDS['config'] = [gws.config.cli.dump, gws.config.cli.dumpmeta, gws.config.cli.prepare, gws.config.cli.test]
COMMANDS['dprocon'] = [gws.ext.action.dprocon.cli.setup]
COMMANDS['gekos'] = [gws.ext.action.gekos.cli.setup]
COMMANDS['georisks'] = [gws.ext.action.georisks.cli.aartelink, gws.ext.action.georisks.cli.export]
COMMANDS['server'] = [gws.server.cli.configure, gws.server.cli.reconfigure, gws.server.cli.reload, gws.server.cli.start, gws.server.cli.stop]



def dispatch(argv):
    parser = argh.ArghParser()

    for ns, fns in COMMANDS.items():
        parser.add_commands(fns, namespace=ns)

    with ch.pretty_errors():
        parser.dispatch(argv)


def main():
    argv = []
    verbose = False
    for a in sys.argv:
        if a == '-v' or a == '--verbose':
            verbose = True
        else:
            argv.append(a)

    gws.log.set_level('DEBUG' if verbose else 'INFO')

    try:
        dispatch(argv[1:])
    except Exception:
        sys.stdout.flush()
        if verbose:
            gws.log.exception()
        sys.exit(1)


if __name__ == '__main__':
    main()
