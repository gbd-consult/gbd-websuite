import sys

import gws
import gws.spec.runtime


def main(argv):
    gws.ensure_system_dirs()

    args, params = parse_args(argv[1:])

    verbose = params.get('v') or params.get('verbose')
    if verbose:
        params.set('logLevel', 'DEBUG')
    elif not params.get('logLevel'):
        params.set('logLevel', 'INFO')

    gws.log.set_level(params.get('logLevel'))

    try:
        specs = gws.spec.runtime.create(params.get('manifest'), read_cache=True, write_cache=True)
    except:
        sys.stdout.flush()
        gws.log.exception()
        return 255

    # all cli command lines are "gws command subcommand --opt1 val --opt2 ...."

    if not args:
        return usage(specs, None, None)

    if len(args) == 1:
        return usage(specs, args[0], None)

    if len(args) > 2:
        return usage(specs, args[0], args[1])

    if params.get('h') or params.get('help'):
        return usage(specs, args[0], args[1])

    # 'gws auth password' => 'authPassword'
    cmd_name = camelize(args[0] + '-' + args[1])
    cmd_desc = command_descriptor(specs, cmd_name, params)
    if not cmd_desc:
        return usage(specs, args[0], args[1])

    try:
        prn('')
        cmd_desc.methodPtr(cmd_desc.request)
        prn('')
    except:
        sys.stdout.flush()
        gws.log.exception()
        return 255


def usage(specs, cmd1, cmd2):
    docs = specs.cli_docs('en')

    s = f'GWS version {specs.version}'
    prn('')
    prn(s)
    prn('~' * len(s))
    prn('')

    show = [d for d in docs if d[0] == cmd1 and d[1] == cmd2]
    if not show:
        show = [d for d in docs if d[0] == cmd1]
    if not show:
        show = docs

    for d in sorted(show):
        prn(d[2])

    return 1


def command_descriptor(specs, cmd_name, params):
    cmd_desc = specs.command_descriptor('cli', cmd_name)
    if not cmd_desc:
        return None

    try:
        cmd_desc.request = specs.read(params, cmd_desc.tArg, strict_mode=False)
    except gws.spec.ReadError:
        return None

    root = gws.create_root_object(specs)
    obj_desc = specs.object_descriptor(cmd_desc.tOwner)
    gws.load_class(obj_desc)
    cmd_desc.methodPtr = getattr(obj_desc.classPtr(), cmd_desc.methodName)

    return cmd_desc


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


def prn(msg):
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
