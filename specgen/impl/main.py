import os
import json
from . import api, normalizer, parser, spec, typescript


def run(source_dir, out_dir, VERSION):
    def _write(p, s):
        s = s.replace('@@VERSION@@', VERSION)
        with open(os.path.join(out_dir, p), 'wt') as fp:
            fp.write(s)

    def _json(r):
        return json.dumps(r, indent=4, sort_keys=True, ensure_ascii=False)

    def _get(dct, keys, default):
        for k in keys:
            if k in dct:
                return dct[k]
        return default

    def _extract_docs(spc):
        doc = {}

        for c in _get(spc, ['types', 'commands'], {}).values():
            doc[c['name']] = c.get('doc', '')
            for p in _get(c, ['props', 'args'], []):
                doc[c['name'] + ':' + p['name']] = p.get('doc', '')

        return doc

    def _merge_docs(spc, what, lang):

        docfile = out_dir + '/lang/' + lang + '.' + what + '.doc.json'
        with open(docfile) as fp:
            docs = json.load(fp)

        for c in _get(spc, ['types', 'commands'], {}).values():
            s = docs.get(c['name'], '')
            if s:
                c['doc'] = s
            for p in _get(c, ['props', 'args'], []):
                s = docs.get(c['name'] + ':' + p['name'], '')
                if s:
                    p['doc'] = s

        _write(lang + '.' + what + '.spec.json', _json(spc))

    def _cli_commands(objects):
        cmds = {}

        for c in objects:
            if c['kind'] == 'clifunc':
                tname = c['command'] + '.' + c['name']
                c['subcommand'] = c['name']
                c['name'] = tname
                cmds[tname] = c

        return cmds

    try:
        objects = parser.parse(source_dir)
    except parser.Error as e:
        print('PARSE ERROR: %s' % e.args[0])
        print('File "%s", line %s' % (e.args[1], e.args[2]))
        print('-' * 40)
        raise

    objects = normalizer.normalize(objects)

    # for p in objects:
    #     print(p)

    config_root = 'gws.common.application.Config'

    conf_spec = {
        'types': spec.generate(objects, [config_root])
    }
    _write('config.spec.json', _json(conf_spec))

    d = _extract_docs(conf_spec)
    _write('config.doc.json', _json(d))

    _merge_docs(conf_spec, 'config', 'de')

    api_methods = api.enum_methods(objects)
    api_types = api.make_spec(objects, api_methods)
    _write('api.spec.json', _json({'types': api_types, 'methods': api_methods}))

    api_types_tree = api.make_spec(objects, api_methods, flatten=False)

    decls, stub = typescript.generate(api_types_tree, api_methods)
    _write('gws-server.api.ts', decls)
    _write('gws-server.base.ts', stub)

    cli_spec = {
        'commands': _cli_commands(objects)
    }

    _write('cli.spec.json', _json(cli_spec))

    d = _extract_docs(cli_spec)
    _write('cli.doc.json', _json(d))

    _merge_docs(cli_spec, 'cli', 'de')
