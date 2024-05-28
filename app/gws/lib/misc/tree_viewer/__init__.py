"""Take a pickled GWS tree and output a browsable HTML page."""

import builtins
import json
import os
import pickle
import sys
import re


def load_pickle(path):
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'builtins':
                return getattr(builtins, name)

            class T:
                KLASS = module + '.' + name

                def __init__(self, *args):
                    pass

                def __setstate__(self, state):
                    if not isinstance(state, dict):
                        state = {'?': state}
                    self.STATE = state

                def __getattr__(self, item):
                    return '?' + item

                def __call__(self, *args, **kwargs):
                    pass

                def __setitem__(self, *args, **kwargs):
                    pass

            return T

    hash_map = {}
    obj_list = []

    def transform(obj):
        if obj is None:
            return obj
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple, set)):
            return [transform(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): transform(v) for k, v in obj.items()}
        if hasattr(obj, 'KLASS'):
            h = hash(obj)
            if h in hash_map:
                return hash_map[h]

            idx = len(obj_list)
            obj_list.append('dummy value')

            cls = obj.KLASS
            if isinstance(obj, type):
                cls = 'CLASS:' + cls

            state = getattr(obj, 'STATE', {})
            if not isinstance(state, dict):
                state = {'state': repr(state)}

            name = '$.' + cls + ':' + str(idx)
            if state.get('uid'):
                name += ':uid=' + str(state['uid'])
            hash_map[h] = name

            d = {str(k): transform(v) for k, v in state.items()}
            d['$'] = name
            obj_list[idx] = d

            return name

        return 'UNKNOWN: ' + repr(obj)

    def flat(val):
        if isinstance(val, str):
            return [val]
        if isinstance(val, (list, tuple, set)):
            return [v for e in val for v in flat(e)]
        if isinstance(val, dict):
            return [v for e in val.values() for v in flat(e)]
        return []

    def make_refs():
        ref_map = {d['$']: d for d in obj_list}
        for d in obj_list:
            for prop, v in d.items():
                if prop == '$':
                    continue
                for v2 in flat(v):
                    if v2 in ref_map and v2 != d['$']:
                        ref_map[v2].setdefault('$REFS', []).append(d['$'] + ' @ ' + prop)

    with open(path, 'rb') as fp:
        tree = Unpickler(fp).load()

    transform(tree)
    make_refs()

    return obj_list


path = sys.argv[1]
obj_list = load_pickle(path)

try:
    mode = sys.argv[2]
except:
    mode = 'json'

path = json.dumps(path)
data = json.dumps(obj_list, indent=4, sort_keys=True).replace('</script>', '<\\/script>')

if mode == 'html':
    cdir = os.path.dirname(__file__)

    with open(cdir + '/css.css') as fp:
        css = f'<style>\n{fp.read()}\n</style>'
    with open(cdir + '/js.js') as fp:
        js = f'<script>\n{fp.read()}\n</script>'

    # for debugging
    # css = f'<link rel="stylesheet" href="{cdir}/css.css">'
    # js = f'<script src="{cdir}/js.js"></script>'

    content = f"""<!doctype html>
        <html>
            <head>
                <meta charset="utf-8">
                {css}
            </head>
            <body>
                <script>PATH={path}</script>
                <script>DATA={data}</script>
                {js}
            </body>
        </html>
    """

    print(content)

if mode == 'json':
    print(data)

if mode == 'bounds':
    obj_map = {d['$']: d for d in obj_list}

    for d in obj_list:
        b = d.get('wgsExtent')
        if not b:
            continue
        x1, y1, x2, y2 = b
        feature = {
            "type": "Feature",
            "properties": {
                "id": d['$'],
                "name": d.get('name') or d.get('title') or '',
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]
                ]
            }
        }
        file_name = re.sub(
            r'\W+',
            '_',
            feature['properties']['name'] + '_' + feature['properties']['id'],
        )
        fc = {
            "type": "FeatureCollection",
            "features": [feature],
        }
        with open(f'{file_name}.geojson', 'w') as fp:
            json.dump(fc, fp, indent=4)
