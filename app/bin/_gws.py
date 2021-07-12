import sys

import gws
import gws.config.error
import gws.server.control
import gws.spec.generator
import gws.spec.runtime

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

_cached_specs = None


def load_specs(params):
    global _cached_specs

    if _cached_specs:
        return _cached_specs

    opts = {
        'manifest_path': params.get('manifest'),
        'cache_path': gws.OBJECT_CACHE_DIR + '/server.spec.json',
    }

    _cached_specs = gws.spec.runtime.from_dict(gws.spec.generator.generate_for_server(opts))
    return _cached_specs


def parse_args(argv):
    args = []
    kwargs = {}
    key = None

    for a in argv:
        if a.startswith('--'):
            key = a[2:]
            kwargs[key] = True
        elif a.startswith('-'):
            key = a[1:]
            kwargs[key] = True
        elif key:
            kwargs[key] = a
            key = None
        else:
            args.append(a)

    return args, gws.Data(kwargs)


def print_error(exc):
    def prn(a):
        if isinstance(a, (list, tuple)):
            for item in a:
                prn(item)
        elif a is not None:
            for s in gws.lines(str(a)):
                print(s)

    prn('-' * 60)

    if isinstance(exc, gws.config.error.ParseError):
        prn('CONFIGURATION PARSE ERROR:')
    elif isinstance(exc, gws.config.error.ConfigError):
        prn('CONFIGURATION ERROR:')
    elif isinstance(exc, gws.config.error.LoadError):
        prn('CONFIGURATION LOAD ERROR:')
    elif isinstance(exc, gws.config.error.MapproxyConfigError):
        prn('MAPPROXY CONFIGURATION ERROR:')
    else:
        prn('SYSTEM ERROR:')

    prn(exc.args)
    prn('-' * 60)


def print_usage_and_fail(ext_type, params):
    for s in load_specs(params).objects('CLI/'):
        print(s)

    print('USAGE', ext_type)
    return 1


def dispatch(ext_type, cmd, params):
    if ext_type == 'server':
        # server commands are special and don't depend on specs
        cli = gws.server.control.Cli()
        fn = getattr(cli, cmd, None)
        if not fn:
            return print_usage_and_fail('server', params)
        return fn(params)

    specs = load_specs(params)
    # e.g. 'gws auth password' => 'authPassword'
    cmd_name = ext_type + cmd[0].upper() + cmd[1:]

    try:
        command_desc = specs.check_command(cmd_name, 'cli', params, strict=False)
    except gws.spec.runtime.Error:
        return print_usage_and_fail(ext_type, params)

    if not command_desc:
        return print_usage_and_fail(ext_type, params)

    object_desc = gws.load_ext(specs, command_desc.class_name)
    handler = object_desc.class_ptr()
    return getattr(handler, command_desc.function_name)(command_desc.params)


def main():
    args, params = parse_args(sys.argv[1:])
    verbose = params.get('v') or params.get('verbose')
    gws.log.set_level('DEBUG' if verbose else 'INFO')

    # all cli actions are "ext_type cmd --opt1 val --opt2 ...."

    if len(args) == 0 or len(args) > 2:
        return print_usage_and_fail(None, params)

    if len(args) == 1:
        return print_usage_and_fail(args[0], params)

    if len(args) == 2 and args[0] == 'help':
        return print_usage_and_fail(args[1], params)

    try:
        return dispatch(args[0], args[1], params)
    except Exception as e:
        sys.stdout.flush()
        print_error(e)
        if verbose:
            gws.log.exception()
        return 255


if __name__ == '__main__':
    sys.exit(main())
