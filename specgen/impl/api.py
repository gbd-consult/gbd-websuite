"""Extract API-specific data from parse results"""

from . import spec


def make_spec(objects, methods, flatten=True):
    types = (
            [m['arg'] for m in methods.values() if 'arg' in m] +
            [m['return'] for m in methods.values() if 'return' in m])
    return spec.generate(objects, types, flatten)


def enum_methods(objects):
    ms = {}

    for p in objects:
        if p['kind'] != 'method':
            continue

        cat = _category(p)
        if not cat:
            continue

        m = {
            'method': p['name'],
            'module': p['module'],
            'doc': p['doc'],
            'category': cat,
            'action': p['module'].split('.')[-1],
        }

        if cat == 'api':
            for c in objects:
                if c['kind'] == 'arg' and c['type'] != 'void' and c['parent_uid'] == p['uid']:
                    m['arg'] = c['type']
                if c['kind'] == 'return' and c['type'] != 'void' and c['parent_uid'] == p['uid']:
                    m['return'] = c['type']

        # convert module/method names to public command identifiers
        # map.api_render_xyz => mapRenderXyz
        # map.http_get_xyz => mapHttpGetXyz

        f = m['method'].split('_')
        if cat == 'api':
            f.pop(0)
        m['cmd'] = m['module'].split('.')[-1] + ''.join(s.title() for s in f)
        ms[m['cmd']] = m

    return ms





def _category(p):
    if p['name'].startswith('api'):
        return 'api'
    if p['name'].startswith('http_get'):
        return 'http_get'
    if p['name'].startswith('http_post'):
        return 'http_post'
    if p['name'].startswith('http'):
        return 'http'

