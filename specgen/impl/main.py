import os
import json
from . import normalizer, parser, spec, typescript


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

    def run(self):
        try:
            self.units = parser.parse(self.source_dir)
        except parser.Error as e:
            print('PARSE ERROR: %s' % e.args[0])
            print('File "%s", line %s' % (e.args[1], e.args[2]))
            print('-' * 40)
            raise


        self.units = normalizer.normalize(self.units)

        ## print(_json(self.units))

        config_root = 'gws.common.application.Config'
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

        # extract docstrings for translations and merge existing translations

        self.extract_and_merge_docs(specs, 'de')
        self.write_specs(specs, 'de')

        # typescript interface and stub files for API methods

        api_methods = [m for m in all_methods.values() if m.category == 'api']
        api_types_tree = _arg_and_return_spec(self.units, api_methods, flatten=False)

        ifaces, stub = typescript.generate(api_types_tree, api_methods)

        self.write('gws-server.api.ts', ifaces)
        self.write('gws-server.base.ts', stub)

    def extract_and_merge_docs(self, specs, lang):
        map = {}

        for name, s in specs.items():
            map[name] = s
            for p in s.get('props', []):
                map[name + ':' + p.name] = p
            for p in s.get('args', []):
                map[name + ':' + p.name] = p

        self.write('doc.json', _json({k: v.doc for k, v in sorted(map.items())}))

        with open(self.out_dir + '/../lang/' + lang + '.json') as fp:
            translations = json.load(fp)

        for k, v in translations.items():
            if k in map:
                map[k].doc = v
            else:
                pass
                ##print('unbound translation: ' + k)

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
