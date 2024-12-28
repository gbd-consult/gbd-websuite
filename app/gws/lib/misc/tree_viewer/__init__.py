"""Take a pickled GWS tree and output a browsable HTML page."""

import builtins
import json
import os
import pickle
import sys


def main():
    cp = ConfigPrinter(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'json')
    out = cp.run()
    print(out)


##

class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'builtins':
            return getattr(builtins, name)

        class T:
            CLASS_NAME = module + '.' + name

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

            def _unpickle(self, *args):
                pass

        return T


class ConfigPrinter:
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.obj_list = []
        self.hash_map = {}

    def run(self):
        self.load_pickle()
        if self.mode == 'html':
            return self.render_html()
        if self.mode == 'json':
            return self.render_json()

    def load_pickle(self):
        with open(self.path, 'rb') as fp:
            tree = Unpickler(fp).load()
        self.transform(tree)
        self.sort_obj_list()
        self.make_refs()

    def transform(self, obj):
        if obj is None:
            return obj
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple, set)):
            return [self.transform(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self.transform(v) for k, v in obj.items()}
        if hasattr(obj, 'CLASS_NAME'):
            return self.transform_object(obj)
        return 'UNKNOWN: ' + repr(obj)

    def transform_object(self, obj):
        h = hash(obj)
        if h in self.hash_map:
            return self.hash_map[h]

        idx = len(self.obj_list)
        self.obj_list.append('DUMMY_VALUE')

        cls = obj.CLASS_NAME
        if isinstance(obj, type):
            cls = 'CLASS.' + cls

        state = getattr(obj, 'STATE', {})
        if not isinstance(state, dict):
            state = {'state': repr(state)}

        ref = '$.' + cls + ':' + str(idx)
        if state.get('uid'):
            ref += ':uid=' + str(state['uid'])
        self.hash_map[h] = ref

        d = {str(k): self.transform(v) for k, v in state.items()}
        d['$'] = ref
        self.obj_list[idx] = d

        return ref

    def make_refs(self):
        def flat(val):
            if isinstance(val, str):
                return [val]
            if isinstance(val, (list, tuple, set)):
                return [v for e in val for v in flat(e)]
            if isinstance(val, dict):
                return [v for e in val.values() for v in flat(e)]
            return []

        ref_map = {d['$']: d for d in self.obj_list}

        for d in self.obj_list:
            for prop, v in d.items():
                if prop == '$' or prop.startswith('~'):
                    continue
                for v2 in flat(v):
                    if v2 in ref_map and v2 != d['$']:
                        ref_map[v2].setdefault('~references', []).append(d['$'] + ' @' + prop)


    def sort_obj_list(self):
        weights = {
            '$.gws.Root': 0,
            '$.gws.base.application.core': 1,
            '$.gws.core': 2,

            '$.gws.base': 20,
            '$.gws.config': 30,
            '$.gws.ext': 40,
            '$.gws.gis': 50,
            '$.gws.lib': 60,
            '$.gws.server': 70,

            '$.gws.plugin': 80,
            '$.gws.test': 90,
            '$.gws.spec': 999,
        }

        def skey(obj):
            ref = obj['$']
            p = ref.split(':')

            cls = p[0]
            try:
                idx = int(p[1])
            except:
                idx = p[1]

            for k, w in weights.items():
                if cls.startswith(k):
                    return w, cls, idx, ref.lower()

            return 100, cls, idx, ref.lower()

        self.obj_list.sort(key=skey)

    DEBUG = False

    def html_resources(self):
        cdir = os.path.dirname(__file__)

        paths = [
            os.path.abspath(cdir + '/../../vendor/jvv/jvv.js'),
            os.path.abspath(cdir + '/../../vendor/jvv/jvv.css'),
            os.path.abspath(cdir + '/tree.css'),
            os.path.abspath(cdir + '/tree.js'),
        ]

        rs = []

        def read(path):
            with open(path, 'rt') as fp:
                return fp.read()

        if self.DEBUG:
            for p in paths:
                p = p.split('lib')[-1]
                if p.endswith('.js'):
                    rs.append(f'<script src="{p}"></script>')
                if p.endswith('.css'):
                    rs.append(f'<link rel="stylesheet" href="{p}">')
        else:
            for p in paths:
                if p.endswith('.js'):
                    rs.append(f'<script>\n' + read(p) + '</script>')
                if p.endswith('.css'):
                    rs.append(f'<style>\n' + read(p) + '</style>')

        return '\n'.join(rs)


    def render_html(self):
        obj_list = json.dumps(self.obj_list, indent=4, sort_keys=True).replace('</script>', '<\\/script>')
        res = self.html_resources()

        return f"""<!doctype html>
            <html>
                <head>
                    <meta charset="utf-8">
                </head>
                <body>
                    <div id="content">
                        <div id="sidebar">
                            <div id="sidebar-top">
                                <div id="sidebar-search-box">
                                    <input>
                                    <button>&times;</button>
                                </div>
                            </div>
                            <div id="sidebar-body"></div>
                        </div>
                        <div id="main"></div>
                    </div>

                    {res}
                    
                    <script id="OBJ_LIST" type="text/plain" defer>
                        {obj_list}
                    </script>
                </body>
            </html>
        """

    def render_json(self):
        return json.dumps(self.obj_list, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
