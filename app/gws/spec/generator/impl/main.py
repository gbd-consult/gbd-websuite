import re
import os
import json
from . import normalizer, parser, spec, typescript, makestubs


def run(source_dir, out_dir, version):
    r = _Runner(source_dir, out_dir, version)
    r.run()


##

class _Runner:
    def __init__(self, source_dir, out_dir, version):
        self.source_dir = source_dir
        self.out_dir = out_dir
        self.version = version
        self.units = []
        self.stubs = {}

    def run(self):
        paths = [p for p in _find_files(self.source_dir, 'py$') if not _ignore_path(p)]

        try:
            self.units, stubs = parser.parse(paths)
        except parser.Error as e:
            print('PARSE ERROR: %s' % e.args[0])
            print('-' * 40)
            raise

        # create /types/__init__.py

        d = self.source_dir + '/types'
        src_path = d + '/__init__.in.py'
        dst_path = d + '/__init__.py'

        src = _read(src_path)
        dst = src + '\n\n' + makestubs.run(stubs)
        _write(dst_path, dst)

        # and parse it
        units, _ = parser.parse([dst_path])
        self.units.extend(units)

        # for u in self.units:
        #     u.dump()

        self.units = normalizer.normalize(self.units)

        config_root = 'gws.base.application.Config'
        all_methods = spec.for_methods(self.units)
        cli_funcs = spec.for_cli_functions(self.units)

        specs = {}

        # create specs for a) types used in configuation

        specs.update(spec.for_types(self.units, [config_root]))

        # ...and b) method argument and return types

        specs.update(_arg_and_return_spec(self.units, all_methods.values(), flatten=True))

        # specs for api and cli methods

        specs.update({'method:' + k: v for k, v in all_methods.items()})
        specs.update({'cli:' + k: v for k, v in cli_funcs.items()})

        self.write_specs(specs)

        # main cli script

        _make_cli_main(cli_funcs, self.source_dir + '/../bin/_gws.in.py')

        # extract docstrings for translations and merge existing translations

        self.extract_and_merge_docs(specs, 'de')
        self.write_specs(specs, 'de')

        # typescript interface and stub files for API methods

        api_methods = [m for m in all_methods.values() if m.category == 'api']
        api_methods.sort(key=lambda m: m.cmd)
        api_types_tree = _arg_and_return_spec(self.units, api_methods, flatten=False)

        ifaces, stub = typescript.generate(api_types_tree, api_methods)

        self.write('gws-server.api.ts', ifaces)
        self.write('gws-server.base.ts', stub)

    # @TODO: should use unflattened types for this

    def extract_and_merge_docs(self, specs, lang):
        dmap = {}

        for name, s in specs.items():
            dmap[name] = s
            for p in s.get('props', []):
                dmap[name + ':' + p.name] = p
            for p in s.get('args', []):
                dmap[name + ':' + p.name] = p

        _kvp_write(os.path.join(self.out_dir, 'doc.txt'), {k: v.doc for k, v in dmap.items()})
        translations = _kvp_read(self.out_dir + '/../lang/' + lang + '.txt')

        for k, v in dmap.items():
            if k in translations:
                dmap[k].doc = translations[k]
            elif dmap[k].doc:
                ## print('not found translation: ' + k)
                pass

        for k, v in translations.items():
            if k not in dmap:
                ## print('unbound translation: ' + k)
                pass

    def write_specs(self, specs, lang=None):
        fname = 'spec.json'
        if lang:
            fname = lang + '.' + fname
        self.write(fname, _json(specs))

    def write(self, path, text):
        text = text.replace('@@VERSION@@', self.version)
        with open(os.path.join(self.out_dir, path), 'wt') as fp:
            fp.write(text)


def _arg_and_return_spec(units, method_specs, flatten):
    types = []

    for m in sorted(method_specs, key=lambda m: m.name):
        if m.arg:
            types.append(m.arg)
        if m.ret:
            types.append(m.ret)

    return spec.for_types(units, types, flatten)


def _json(r):
    return json.dumps(r, indent=4, sort_keys=True, ensure_ascii=False, default=vars)


def _kvp_read(path):
    ls = []
    for s in _read(path).strip().splitlines():
        s = s.strip()
        if not s or s.startswith('#') or '=' not in s:
            continue
        s = s.split('=')
        ls.append([s[0].strip(), s[1].strip()])
    return {k: v for k, v in sorted(ls)}


def _kvp_write(path, d):
    ls = []
    for k, v in sorted(d.items()):
        ls.append(k + ' = ' + v)
    _write(path, '\n'.join(ls))


def _find_files(dirname, pattern):
    for fname in os.listdir(dirname):
        if fname.startswith('.'):
            continue

        path = os.path.join(dirname, fname)

        if os.path.isdir(path):
            yield from _find_files(path, pattern)
            continue

        if re.search(pattern, fname):
            yield path


def _ignore_path(p):
    return 'types/' in p or p.startswith('___') or p.endswith('.in.py')


def _make_cli_main(cli_funcs, path):
    cmds = {}
    mods = set()

    for p in cli_funcs.values():
        if p.command not in cmds:
            cmds[p.command] = []
        cmds[p.command].append(p.module + '.' + p.subcommand)
        mods.add(p.module)

    py = []

    for m in sorted(mods):
        py.append(f'import {m}')
    py.append('')

    for cmd, fns in sorted(cmds.items()):
        py.append('COMMANDS[%r] = [%s]' % (cmd, ', '.join(sorted(fns))))
    py.append('')

    with open(path) as fp:
        text = fp.read()

    text = text.replace('#@COMMANDS', '\n'.join(py))

    path = path.replace('.in', '')
    with open(path, 'w') as fp:
        fp.write(text)


def _read(path):
    with open(path, encoding='utf8') as fp:
        return fp.read()


def _write(path, txt):
    with open(path, 'wt', encoding='utf8') as fp:
        return fp.write(txt)
