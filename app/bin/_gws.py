import sys

import gws
import gws.spec.runtime
import gws.base.action
import gws.lib.cli as cli


def main(args):
    try:
        return main2(args)
    except gws.ConfigurationError:
        gws.log.exception()
        cli.error('STOP')
        return 3
    except Exception:
        cli.error('INTERNAL ERROR:')
        gws.log.exception()
        return 4


def main2(args):
    gws.u.ensure_system_dirs()

    gws.log.set_level(gws.env.GWS_LOG_LEVEL or 'INFO')
    if args.pop('v', None) or args.pop('verbose', None):
        gws.log.set_level('DEBUG')

    manifest_path = real_manifest_path(args.get('manifest', None))
    if manifest_path and 'manifest' in args:
        args['manifest'] = manifest_path

    # specs are fast enough, don't bother with caching for now
    specs = gws.spec.runtime.create(manifest_path or '', read_cache=False, write_cache=False)

    # all cli command lines are "gws command subcommand -opt1 val1 -opt2 val2 ...."
    # command + sub are translated to a camelized method name:  'gws auth password' => 'authPassword'
    # argument names are camelized as well

    args.pop(0, None)
    cmd1 = args.pop(1, '')
    cmd2 = args.pop(2, '')

    if not cmd1:
        print_usage(specs, None)
        return 0

    if args.pop('h', None) or args.pop('help', None):
        print_usage(specs, cmd1)
        return 0

    if not cmd2 or args.get(3):
        cli.error('invalid arguments, try "gws -h" for help')
        return 1

    root = gws.create_root(specs)

    try:
        fn, request = gws.base.action.parse_cli_request(
            root,
            command_category=gws.CommandCategory.cli,
            command_name=camelize(cmd1 + '-' + cmd2),
            params={camelize(key): val for key, val in args.items()},
            read_options={gws.SpecReadOption.caseInsensitive, gws.SpecReadOption.convertValues}
        )
    except gws.NotFoundError:
        cli.error('command not found, try "gws -h" for help')
        return 1
    except gws.ForbiddenError:
        cli.error('command forbidden')
        return 1
    except gws.BadRequestError:
        cli.error('invalid arguments, try "gws -h" for help')
        return 1

    cli.info('')
    fn(request)
    cli.info('')

    return 0


def print_usage(specs, cmd1):
    tab = ' ' * 4
    me = 'gws'

    def banner():
        s = f'GWS version {specs.version}'
        cli.info('\n' + s + '\n' + ('~' * len(s)) + '\n')

    def type_name(t):
        if t == 'bool':
            return ''
        if ',' in t:
            t = t.split(',')
            if t[0] == 'LIST':
                return '<' + t[1] + ',' + t[1] + '...>'
        return '<' + t + '>'

    banner()

    cs = specs.cli_commands('en')

    # gws -h
    if not cmd1 or all(c.cmd1 != cmd1 for c in cs):
        rows = [
            [f'{me} {c.cmd1} {c.cmd2}', f'- {c.doc}']
            for c in cs
        ]
        cli.info(cli.text_table(rows, delim=tab))
        cli.info(f'\nTry "{me} <command> -h" for more info.\n')
        return

    # gws command -h
    for c in cs:
        if c.cmd1 == cmd1:
            cli.info(f'{me} {c.cmd1} {c.cmd2}')
            cli.info(f'{tab}{c.doc}')

            opts = []
            for a in c.args:
                doc = a.doc + ('' if a.hasDefault else ' (required)')
                opts.append([f'{tab}', f'-{a.name}', f'{type_name(a.type)}', f'- {doc}'])
            if opts:
                cli.info(f'{tab}Options:')
                cli.info(cli.text_table(opts, delim=tab))
            cli.info('')


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


# from config.loader

_DEFAULT_MANIFEST_PATHS = [
    '/data/MANIFEST.json',
]


def real_manifest_path(manifest_path):
    p = manifest_path or gws.env.GWS_MANIFEST
    if p:
        return p
    for p in _DEFAULT_MANIFEST_PATHS:
        if gws.u.is_file(p):
            return p


if __name__ == '__main__':
    sys.exit(main(cli.parse_args(sys.argv)) or 0)
