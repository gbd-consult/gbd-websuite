import sys

import argh

import gws
import gws.tools.misc as misc
import gws.tools.clihelpers as ch

misc.ensure_dir(gws.MAPPROXY_CACHE_DIR)
misc.ensure_dir(gws.WEB_CACHE_DIR)
misc.ensure_dir(gws.OBJECT_CACHE_DIR)
misc.ensure_dir(gws.NET_CACHE_DIR)
misc.ensure_dir(gws.CONFIG_DIR)
misc.ensure_dir(gws.LOG_DIR)
misc.ensure_dir(gws.MISC_DIR)
misc.ensure_dir(gws.PRINT_DIR)
misc.ensure_dir(gws.SERVER_DIR)


def dispatch(argv):
    parser = argh.ArghParser()

    for ns, fns in ch.find_commands().items():
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
