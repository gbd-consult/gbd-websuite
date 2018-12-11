import os
import json
from . import api, normalizer, parser, spec, typescript


def _json(r):
    return json.dumps(r, indent=4, sort_keys=True)


def run(source_dir, out_dir, VERSION):

    def _write(p, s):
        s = s.replace('@@VERSION@@', VERSION)
        with open(os.path.join(out_dir, p), 'wt') as fp:
            fp.write(s)

    try:
        objects = parser.parse(source_dir)
    except parser.Error as e:
        print('PARSE ERROR: %s' % e.args[0])
        print('File "%s", line %s' % (e.args[1], e.args[2]))
        print('-' * 40)
        raise

    # for p in objects:
    #     print(p)
    # print('-' * 40)

    normalizer.normalize(objects)

    config_root = 'gws.common.application.Config'

    types = spec.generate(objects, [config_root])
    _write('config.spec.json', _json({'types': types}))

    methods = api.enum_methods(objects)
    types = api.make_spec(objects, methods, keep_extends=False)
    _write('api.spec.json', _json({'types': types, 'methods': methods}))

    types = api.make_spec(objects, methods, keep_extends=True)
    ## _write('api-debug.spec.json', _json({'types': types, 'methods': methods}))

    decls, stub = typescript.generate(types, methods)
    _write('gws-server.api.ts', decls)
    _write('gws-server.base.ts', stub)
    ###_write('gws-server.json', json.dumps({'version': VERSION}))

    cli_funcs = [p for p in objects if p['kind'] == 'clifunc']
    _write('cli.spec.json', _json(cli_funcs))
