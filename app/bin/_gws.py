import sys

import gws
import gws.spec.runtime
import gws.base.action.dispatcher
import gws.base.web.error
import gws.lib.cli as cli


def main(args):
    try:
        return main2(args)
    except:
        cli.error('INTERNAL ERROR:')
        gws.log.exception()
        return 2


def main2(args):
    gws.ensure_system_dirs()

    gws.log.set_level(gws.env.GWS_LOG_LEVEL or 'INFO')
    if args.pop('v', None) or args.pop('verbose', None):
        gws.log.set_level('DEBUG')

    manifest = args.pop('manifest', None) or gws.env.GWS_MANIFEST

    # specs are fast enough, don't bother with caching for now
    specs = gws.spec.runtime.create(manifest, read_cache=False, write_cache=False)

    # all cli command lines are "gws command subcommand -opt1 val1 -opt2 val2 ...."
    # cmd1 + cmd2 are translated to a camelized method name:  'gws auth password' => 'authPassword'
    # argument names should be camilized as well

    args.pop(0, None)
    cmd1 = args.pop(1, '')
    cmd2 = args.pop(2, '')

    if not cmd1:
        print_usage(specs, None, None)
        return 0

    if args.pop('h', None) or args.pop('help', None):
        print_usage(specs, cmd1, cmd2)
        return 0

    if not cmd2 or args.get(3):
        cli.error('invalid arguments, try "gws -h" for help')
        return 1

    root = gws.create_root_object(specs)

    try:
        fn, request = gws.base.action.dispatcher.dispatch(
            root,
            command_category='cli',
            command_name=camelize(cmd1 + '-' + cmd2),
            params={camelize(key): val for key, val in args.items()},
            read_options={'case_insensitive', 'convert_values', 'ignore_extra_props'}
        )
    except gws.base.action.CommandNotFound:
        cli.error('command not found, try "gws -h" for help')
        return 1
    except gws.base.action.CommandForbidden:
        cli.error('command forbidden')
        return 1
    except gws.base.action.BadRequest:
        cli.error('invalid arguments, try "gws -h" for help')
        return 1

    cli.info('')
    fn(request)
    cli.info('')

    return 0


def print_usage(specs, cmd1, cmd2):
    tab = ' ' * 4
    me = 'gws'

    def banner():
        s = f'GWS version {specs.version}'
        cli.info('\n' + s + '\n' + ('~' * len(s)) + '\n')

    def columns(lines, align='<'):
        col1, col2 = zip(*lines)
        maxlen = max(len(a) for a in col1)
        return '\n'.join(f'{a:{align}{maxlen}s}{b}' for a, b in zip(col1, col2))

    def options(cmd):
        opts = []
        for a in cmd.args:
            if not a.hasDefault:
                opts.append([f'{tab}{tab}--{a.name}', f'{tab}<{a.doc}> (required)'])
        for a in cmd.args:
            if a.hasDefault:
                opts.append([f'{tab}{tab}--{a.name}', f'{tab}<{a.doc}>'])
        if opts:
            cli.info(f'{tab}Options:')
            cli.info(columns(opts, align='<'))
            cli.info('')

    banner()

    cs = specs.cli_commands('en')

    # gws -h
    if not cmd1 or all(c.cmd1 != cmd1 for c in cs):
        cli.info(columns([f'{tab}{me} {c.cmd1} {c.cmd2}', '- ' + c.doc] for c in cs))
        cli.info('\nTry "{me} <command> -h" for more info.\n')
        return

    # gws command -h
    for c in cs:
        if c.cmd1 == cmd1:
            cli.info(f'{me} {c.cmd1} {c.cmd2} - {c.doc}\n')
            options(c)


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


if __name__ == '__main__':
    sys.exit(main(cli.parse_args(sys.argv)) or 0)
