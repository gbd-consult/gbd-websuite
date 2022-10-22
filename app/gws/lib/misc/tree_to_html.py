"""Take a pickled GWS tree and output a browsable HTML page."""

CSS = r"""
html, body {
    font: 14px Consolas, monospace;
    box-sizing: border-box;
    background: black;
    color: white;
    
}

#main {
    position: fixed;
    left: 0;
    top: 0;
    right: 0;
    bottom: 60px;
    display: flex;
    overflow-x: scroll;
    background: #151520;
    color: #ADAEAD;
}

#main div {
    max-height: 100%;
    overflow: scroll;
    white-space: pre;
    padding: 5px;
    flex: 1;
    min-width: 450px;
    line-height: 1.5;
}

#main div b {
    font-weight: 500;
    color: #b67b41;
    padding: 3px;
}

#main div i {
    font-style: normal;
    background: yellow;
    color: black;
}

a {
    color: #7D89B5;
    cursor: pointer;
}

a:hover {
    text-decoration: underline;
}

#search {
    position: fixed;
    left: 10px;
    bottom: 10px;
    width: 370px;
    height: 30px;
    background: none;
    color: #ADAEAD;
    padding: 0 10px;
    border: none;
}

#search_clear {
    position: fixed;
    left: 405px;
    bottom: 10px;
    width: 30px;
    height: 30px;
    border: none;
    background: none;
    color: #ADAEAD;
    
}

#path {
    position: fixed;
    right: 10px;
    bottom: 10px;
    color: #888;
}
"""

JS = r"""
let $ = s => document.querySelector(s);

function htmlize(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function regexQuote(str) {
    return str.replace(
        /[.*+?^${}()|[\]\\-]/g,
        s => '\\x' + s.charCodeAt(0).toString(16));
}

function getSearch() {
    return ($("#search").value || '').trim();
}

function addColumn() {
    let d = document.createElement('div');
    $('#main').appendChild(d);
    $('#main').scrollLeft = 9999999;
    return d;
}

function findColumn(el) {
    while (1) {
        if (!el)
            return addColumn();
        if (el.parentNode && el.parentNode.id === 'main')
            return el;
        el = el.parentNode;
    }
}

function formatString(val, isKey) {
    let rel = (!isKey && val.startsWith('$.')) ? val : null;
    let html = htmlize(val);
    let search = getSearch();
    
    if (search)
        html = html.replaceAll(search, '<i>$&</i>');
    if (rel)
        html = '<a rel=' + rel + '>' + html + '</a>';
    if (isKey)
        html = '<b>' + html + '</b>';

    return html;
}

function formatObject(obj) {
    if (typeof obj === 'string')
        return formatString(obj, false);

    if (!obj || typeof obj !== 'object') 
        return obj;
    
    if (Array.isArray(obj))
        return obj.map(x => formatObject(x));
    
    let r = {};
    for (let [k, v] of Object.entries(obj))
        r[formatString(k, true)] = formatObject(v);
    return r;
}     

function showColumn(obj, srcColumn) {
    let main = $("#main");
    while (main.lastChild && main.lastChild !== srcColumn)
        main.removeChild(main.lastChild);
    addColumn().innerHTML = JSON.stringify(formatObject(obj), null, 4);
}

function showObject(ref, srcColumn) {
    location.hash = ref;
    let obj = DATA.find(x => x.$ === ref) || 'object not found';
    showColumn(obj, srcColumn);
}

function showList() {
    let data = DATA;
    let search = getSearch();
    if (search) 
        data = data.filter(x => JSON.stringify(x).toLowerCase().includes(search));
    location.hash = '';
    showColumn(data.map(x => x.$));
}

window.addEventListener('load', init);

function init() {
    $("#path").innerText = PATH;

    let h = location.hash;

    showList();
    
    if (h)
        showObject(h.slice(1), $('#main').lastChild);

    document.body.addEventListener('click', function (evt) {
        let h = evt.target.rel;
        if (h)
            showObject(h.split(' @ ')[0], findColumn(evt.target));
    });

    $('#search').addEventListener('input', function (evt) {
        showList();
    });

    $('#search_clear').addEventListener('click', function (evt) {
        $('#search').value = '';
        showList();
    });
    
}
"""

HTML = r"""
<div id="main"></div>
<input id="search" placeholder="Search">
<button id="search_clear">&times;</button>
<div id="path"></div>
"""

import sys
import pickle
import json
import builtins


def load_pickle(path):
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'builtins':
                return getattr(builtins, name)

            class T:
                KLASS = module + '.' + name

                def __setstate__(self, state):
                    if not isinstance(state, dict):
                        state = {'?': state}
                    self.STATE = state

                def __getattr__(self, item):
                    return '?' + item

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
            return {k: transform(v) for k, v in obj.items()}
        if hasattr(obj, 'KLASS'):
            h = hash(obj)
            if h in hash_map:
                return hash_map[h]

            idx = len(obj_list)
            obj_list.append('dummy value')

            cls = obj.KLASS
            if isinstance(obj, type):
                cls = 'CLASS:' + cls

            name = '$.' + cls + ':' + str(idx)
            hash_map[h] = name

            state = getattr(obj, 'STATE', {})
            if isinstance(state, dict):
                d = {k: transform(v) for k, v in state.items()}
            else:
                d = {'state': state}
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

path = json.dumps(path)
data = json.dumps(obj_list, indent=4, sort_keys=True).replace('</script>', '<\\/script>')

content = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
{CSS}
</style>
<script>PATH={path}</script>
<script>DATA={data}</script>
<script>{JS}</script>
</head>
<body>
{HTML}
</body>
</html>
"""


print(content)
