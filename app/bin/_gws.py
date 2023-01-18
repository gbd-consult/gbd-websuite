import sys

import gws
import gws.spec.runtime
import gws.base.action.dispatcher
import gws.base.web.error


def main(argv):
    gws.ensure_system_dirs()

    # all cli command lines are "gws command subcommand --opt1 val1 --opt2 val2 ...."
    # cmd1 + cmd2 are translated to a camelized method name:  'gws auth password' => 'authPassword'

    args, params = parse_args(argv[1:])

    cmd1 = args.pop(0) if args else None
    cmd2 = args.pop(0) if args else None

    verbose = params.get('v') or params.get('verbose')
    if verbose:
        params['logLevel'] = 'DEBUG'
    elif not params.get('logLevel'):
        params['logLevel'] = 'INFO'

    gws.log.set_level(params.get('logLevel'))

    try:
        specs = gws.spec.runtime.create(
            params.get('manifest'),
            read_cache=not params.get('noSpecCache'),
            write_cache=True
        )
    except:
        sys.stdout.flush()
        gws.log.exception()
        return 255

    if params.get('h') or params.get('help'):
        return usage(specs, cmd1, cmd2)

    if not cmd1:
        return usage(specs, None, None)

    if len(args) > 0:
        return error('invalid command')

    root = gws.create_root_object(specs)

    try:
        fn, request = gws.base.action.dispatcher.dispatch(
            root,
            command_category='cli',
            command_name=camelize(cmd1 + '-' + cmd2),
            params=params,
            read_options={'case_insensitive', 'convert_values', 'ignore_extra_props'}
        )
    except gws.base.web.error.NotFound:
        return error('command not found')
    except gws.base.web.error.BadRequest:
        return error('invalid arguments')
    except:
        error('fatal error')
        gws.log.exception()
        return 255

    try:
        writeln()
        fn(request)
        writeln()
    except:
        gws.log.exception()
        return 255


def usage(specs, cmd1, cmd2):
    s = f'GWS version {specs.version}'
    writeln('\n' + s + '\n' + ('~' * len(s)) + '\n')

    def columns(lines, align='<'):
        col1, col2 = zip(*lines)
        maxlen = max(len(a) for a in col1)
        return '\n'.join(
            f'{a:{align}{maxlen}s}{b}'
            for a, b in zip(col1, col2))

    def commands():
        clist = specs.cli_commands('en')
        if not cmd1:
            return clist
        clist1 = [c for c in clist if c.cmd1 == cmd1]
        if not clist1:
            return clist
        if not cmd2:
            return clist1
        clist2 = [c for c in clist1 if c.cmd2 == cmd2]
        if len(clist2) != 1:
            return clist1
        return clist2

    cs = commands()
    tab = ' ' * 4

    if len(cs) == 1:
        one = cs[0]
        writeln(f'gws {one.cmd1} {one.cmd2}\n\n{one.doc}')
        opts = []
        for a in one.args:
            if not a.hasDefault:
                opts.append([f'{tab}--{a.name}', f' <{a.doc}> (required)'])
        for a in one.args:
            if a.hasDefault:
                opts.append([f'{tab}--{a.name}', f' <{a.doc}>'])
        if opts:
            writeln('\nOptions:\n\n' + columns(opts, align='>'))
    else:
        writeln(columns(
            [f'{tab}gws {c.cmd1} {c.cmd2}', '  -  ' + c.doc]
            for c in cs
        ))

    writeln()
    return 1


def error(msg):
    writeln()
    writeln(f'ERROR: {msg}')
    writeln(f'Try gws -h for help')
    writeln()
    return 255


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

    return args, kwargs


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


def writeln(msg=''):
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
