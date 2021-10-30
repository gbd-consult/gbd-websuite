import sys

import gws
import gws.spec.runtime

gws.ensure_system_dirs()


def prn(msg):
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


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


def print_usage_and_fail(specs, ext_type, cmd):
    docs = specs.cli_docs('en')

    prn(f'\nGWS version {gws.VERSION}\n')

    disp = [d for d in docs if d[0] == ext_type and d[1] == cmd]
    if not disp:
        disp = [d for d in docs if d[0] == ext_type]
    if not disp:
        disp = docs

    for d in sorted(disp):
        prn(d[2])

    return 1


def dispatch(root, ext_type, cmd, params):
    # e.g. 'gws auth password' => 'authPassword'
    cmd_name = camelize(ext_type + '-' + cmd)

    try:
        command_desc = root.specs.parse_command(cmd_name, 'cli', params, with_strict_mode=False)
    except gws.spec.runtime.Error:
        return print_usage_and_fail(root.specs, ext_type, cmd)

    if not command_desc:
        return print_usage_and_fail(root.specs, ext_type, cmd)

    handler = root.create_object(command_desc.class_name)
    prn('')
    res = getattr(handler, command_desc.function_name)(command_desc.params)
    prn('')
    return res


def main():
    args, params = parse_args(sys.argv[1:])

    verbose = params.get('v') or params.get('verbose')
    if verbose:
        params.set('logLevel', 'DEBUG')
    elif not params.get('logLevel'):
        params.set('logLevel', 'INFO')

    gws.log.set_level(params.get('logLevel'))

    # all cli command lines are "gws ext_type command_name --opt1 val --opt2 ...."

    try:
        if args[0] == 'spec':
            gws.spec.runtime.create_and_store(params.get('manifest'))
            return 0
        else:
            specs = gws.spec.runtime.load(params.get('manifest'))
    except:
        sys.stdout.flush()
        gws.log.exception()
        return 255

    if len(args) == 0:
        return print_usage_and_fail(specs, None, None)

    if len(args) == 1:
        return print_usage_and_fail(specs, args[0], None)

    if len(args) > 2:
        return print_usage_and_fail(specs, args[0], args[1])

    if params.get('h') or params.get('help'):
        return print_usage_and_fail(specs, args[0], args[1])

    try:
        return dispatch(gws.create_root_object(specs), args[0], args[1], params)
    except:
        sys.stdout.flush()
        gws.log.exception()
        return 255


if __name__ == '__main__':
    sys.exit(main() or 0)
