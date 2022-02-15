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

#@COMMANDS


def dispatch(argv):
    parser = argh.ArghParser()

    for ns, fns in COMMANDS.items():
        parser.add_commands(fns, namespace=ns)

    with ch.pretty_errors():
        parser.dispatch(argv)


def main():
    argv = []
    verbose = False
    n = 0
    while n < len(sys.argv):
        a = sys.argv[n]
        if a == '-v' or a == '--verbose':
            verbose = True
            n += 1
            continue
        # --manifest is accepted, but ignored by now
        if a == '--manifest':
            n += 2
            continue
        argv.append(a)
        n += 1

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
