import sys

import gws
import gws.config
import gws.server.control
import gws.spec.generator
import gws.spec.runtime

gws.ensure_system_dirs()


def spec_runtime(params):
    return gws.spec.runtime.create(params.get('manifest'), with_cache=not params.get('skipSpecCache'))


def camelize(p):
    parts = []
    for s in p.split('-'):
        s = s.strip()
        if s:
            parts.append(s[0].upper() + s[1:])
    if not parts:
        return ''
    s = ''.join(parts)
    return s[0].lower() + s[1:]


def parse_args(argv):
    args = []
    kwargs = {}
    key = None

    for a in argv:
        if a.startswith('--'):
            key = camelize(a[2:])
            kwargs[key] = True
        elif a.startswith('-'):
            key = camelize(a[1:])
            kwargs[key] = True
        elif key:
            kwargs[key] = a
            key = None
        else:
            args.append(a)

    return args, gws.Data(kwargs)


def print_usage_and_fail(ext_type, cmd, params):
    docs = spec_runtime(params).cli_docs('en')

    print('')
    print(f'GWS version {gws.VERSION}')
    print('')

    disp = [d for d in docs if d[0] == ext_type and d[1] == cmd]
    if not disp:
        disp = [d for d in docs if d[0] == ext_type]
    if not disp:
        disp = docs

    for d in sorted(disp):
        print(d[2])

    return 1


def dispatch(ext_type, cmd, params):
    if ext_type == 'server' and cmd == 'stop':
        # skip the spec stuff when stopping the server
        cli = gws.server.control.Cli()
        return cli.stop(params)

    specs = spec_runtime(params)
    # e.g. 'gws auth password' => 'authPassword'
    cmd_name = camelize(ext_type + '-' + cmd)

    try:
        command_desc = specs.check_command(cmd_name, 'cli', params, strict=False)
    except gws.spec.runtime.Error:
        return print_usage_and_fail(ext_type, cmd, params)

    if not command_desc:
        return print_usage_and_fail(ext_type, cmd, params)

    object_desc = gws.load_ext(specs, command_desc.class_name)
    handler = object_desc.class_ptr()
    return getattr(handler, command_desc.function_name)(command_desc.params)


def main():
    args, params = parse_args(sys.argv[1:])

    verbose = params.get('v') or params.get('verbose')
    if verbose:
        params.set('logLevel', 'DEBUG')
    elif not params.get('logLevel'):
        params.set('logLevel', 'INFO')

    gws.log.set_level(params.get('logLevel'))

    # all cli command lines are "gws ext_type command_name --opt1 val --opt2 ...."

    if len(args) == 0:
        return print_usage_and_fail(None, None, params)

    if len(args) == 1:
        return print_usage_and_fail(args[0], None, params)

    if len(args) > 2:
        return print_usage_and_fail(args[0], args[1], params)

    if params.get('h') or params.get('help'):
        return print_usage_and_fail(args[0], args[1], params)

    try:
        return dispatch(args[0], args[1], params)
    except Exception as exc:
        sys.stdout.flush()
        gws.log.exception()
        gws.exit(255)


if __name__ == '__main__':
    sys.exit(main())
