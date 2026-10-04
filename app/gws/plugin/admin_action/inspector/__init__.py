"""Object inspector: an html page to inspect the object tree.

The page lists all tree nodes, shows the properties of the selected object and
lets the user follow references to other objects. It is served by the
``adminInspector`` command of the ``admin`` action.

An object is addressed by a path of ``/``-separated, url-quoted segments.
The first segment is the uid of a tree node, or empty for the root.
Further segments are attribute names, dict keys or list indexes.

Values are shown as primitives, as links to tree nodes and other objects,
or, for dicts, lists and ``gws.Data``, expanded inline up to a fixed depth.

The search walks the whole object graph from the root, breadth first, and lists
objects with a primitive property containing the search text (``text``),
or a specific property containing it (``prop=text``). The search is case-insensitive
and the number of results is limited.

Example::

    /_/adminInspector?path=
    /_/adminInspector?path=<node uid>/config/layers/0
    /_/adminInspector?search=title=roads
"""

import collections
import json
import os
import re
import types
import urllib.parse

import gws
import gws.lib.mime

_DIR = os.path.dirname(__file__)
_ASSETS = {
    'page.js': (f'{_DIR}/page.js', gws.lib.mime.JS),
    'page.css': (f'{_DIR}/page.css', gws.lib.mime.CSS),
}
_URL = f'{gws.c.SERVER_ENDPOINT}/adminInspector?path='
_MAX_COLLECTION_DEPTH = 3
_MAX_REPR_LENGTH = 500
_MAX_SEARCH_RESULTS = 500

_OPAQUE_TYPES = (
    type,
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.MethodWrapperType,
)


def get_content(root: gws.Root, path: str, search: str) -> gws.ContentResponse:
    """Render the inspector page for an object path, or return a page asset.

    Args:
        root: The configuration root.
        path: Object path, or the name of an asset (``page.js``, ``page.css``).
        search: Optional search text, ``text`` or ``prop=text``.

    Returns:
        The rendered page or the asset.

    Raises:
        gws.NotFoundError: If the path cannot be resolved.
    """
    path = path or ''
    search = (search or '').strip()

    if path in _ASSETS:
        p, mime_type = _ASSETS[path]
        return gws.ContentResponse(contentPath=p, mimeType=mime_type)

    obj = _resolve(root, _split(path))

    config = {
        'url': _URL,
        'path': path,
        'selectedUid': _split(path)[0],
        'crumbs': _crumbs(root, path),
        'nodes': [_node_entry(node) for node in root.nodes],
        'search': search,
        'results': _search(root, search) if search else None,
        'maxResults': _MAX_SEARCH_RESULTS,
        'label': _label(obj),
        'props': [_value(root, v, _join(path, k), 0) | {'key': str(k)} for k, v in _entries(obj)],
    }

    tpl = root.app.templateMgr.template_from_path(f'{_DIR}/page.cx.html')
    args = {
        'url': _URL,
        'version': max(int(os.path.getmtime(p)) for p, _ in _ASSETS.values()),
        'configJson': json.dumps(config).replace('<', '\\u003c'),
    }
    return tpl.render(gws.TemplateRenderInput(args=args))


##


def _split(path: str) -> list[str]:
    return [urllib.parse.unquote(s) for s in path.split('/')]


def _join(path: str, key) -> str:
    return path + '/' + urllib.parse.quote(str(key), safe='')


def _node_path(node: gws.Node) -> str:
    return urllib.parse.quote(node.uid, safe='')


def _resolve(root: gws.Root, segs: list[str]):
    """Return the object addressed by path segments."""
    if segs[0]:
        obj = root.uidMap.get(segs[0])
        if obj is None:
            raise gws.NotFoundError(f'object {segs[0]!r} not found')
    else:
        obj = root

    for s in segs[1:]:
        obj = _get(obj, _key(obj, s))

    return obj


def _key(obj, seg: str):
    """Convert a path segment into a key of an object (dict key, list index or attribute name)."""
    if isinstance(obj, dict):
        for k in obj:
            if str(k) == seg:
                return k
    elif _is_list(obj):
        if seg.isdigit() and int(seg) < len(obj):
            return int(seg)
    elif _is_object(obj):
        if seg in vars(obj):
            return seg
    raise gws.NotFoundError(f'{seg!r} not found in {_label(obj)}')


def _get(obj, key):
    """Return a child of an object by key."""
    if isinstance(obj, (set, frozenset)):
        return list(obj)[key]
    if isinstance(obj, (dict, list, tuple)):
        return obj[key]
    return vars(obj)[key]


def _entries(obj) -> list[tuple]:
    """Return the (key, value) pairs of a dict, list or object."""
    if isinstance(obj, dict):
        return list(obj.items())
    if _is_list(obj):
        return list(enumerate(obj))
    if _is_object(obj):
        return sorted(vars(obj).items())
    return []


##


def _value(root: gws.Root, val, path: str, depth: int) -> dict:
    """Describe a value for the page, expanding collections up to a fixed depth."""
    if _is_primitive(val):
        return {'kind': 'primitive', 'baseType': _base_type(val), 'value': _primitive_str(val)}

    if val is root:
        return {'kind': 'object', 'label': _format(val), 'path': ''}

    if _is_tree_node(root, val):
        return {'kind': 'object', 'label': _format(val), 'path': _node_path(val)}

    if (isinstance(val, (dict, gws.Data)) or _is_list(val)) and depth < _MAX_COLLECTION_DEPTH:
        return {
            'kind': 'collection',
            'label': _label(val),
            'path': path,
            'items': [_value(root, v, _join(path, k), depth + 1) | {'key': str(k)} for k, v in _entries(val)],
        }

    if _is_object(val) or _is_list(val):
        return {'kind': 'object', 'label': _label(val), 'path': path}

    return {'kind': 'other', 'value': _repr(val)}


def _search(root: gws.Root, query: str) -> list[dict]:
    """Walk the object graph breadth first and return the objects matching the query."""
    m = re.match(r'^([^=\s]+)\s*=(.*)$', query)
    prop, text = (m.group(1), m.group(2).strip()) if m else (None, query)
    text = text.lower()

    results = []
    found = set()
    seen = set()
    queue = collections.deque([(root, '', root, '')])

    while queue and len(results) < _MAX_SEARCH_RESULTS:
        obj, path, owner, owner_path = queue.popleft()
        if id(obj) in seen:
            continue
        seen.add(id(obj))

        if obj is root or (_is_object(obj) and not isinstance(obj, (dict, gws.Data))):
            owner, owner_path = obj, path

        try:
            entries = _entries(obj)
        except Exception:
            continue

        for k, v in entries:
            if owner_path not in found and (prop is None or str(k) == prop) and not _is_list(obj) and _match(v, text) is not None:
                found.add(owner_path)
                results.append({'path': owner_path, 'label': _format(owner)})
            if _is_primitive(v) or v is root:
                continue
            if _is_tree_node(root, v):
                queue.append((v, _node_path(v), owner, owner_path))
            elif _is_object(v) or _is_list(v):
                queue.append((v, _join(path, k), owner, owner_path))

    return results


def _match(val, text: str):
    """Return the string form of a primitive value, or of a list item, that contains the text, or ``None``."""
    if _is_primitive(val):
        s = _primitive_str(val)
        return s if text in s.lower() else None
    if _is_list(val):
        for v in val:
            if _is_primitive(v):
                s = _primitive_str(v)
                if text in s.lower():
                    return s


def _node_entry(node: gws.Node) -> dict:
    return {
        'uid': node.uid,
        'path': _node_path(node),
        'label': _format(node),
    }


def _crumbs(root: gws.Root, path: str) -> list[dict]:
    """Return breadcrumbs for a path: the root, the parent chain of the node and the path segments."""
    segs = _split(path)
    crumbs = []

    if segs[0]:
        node = root.uidMap.get(segs[0])
        while _is_tree_node(root, node):
            crumbs.insert(0, {'label': _format(node), 'path': _node_path(node)})
            node = vars(node).get('parent')

    crumbs.insert(0, {'label': _format(root), 'path': ''})

    p = _node_path(root.uidMap[segs[0]]) if segs[0] else ''
    for s in segs[1:]:
        p = _join(p, s)
        crumbs.append({'label': s, 'path': p})

    return crumbs


def _label(val) -> str:
    """Return a short label for a value, with the size for collections."""
    if isinstance(val, dict):
        return f'dict[{len(val)}]'
    if _is_list(val):
        return f'{type(val).__name__}[{len(val)}]'
    if isinstance(val, gws.Data):
        return f'{type(val).__name__}[{len(vars(val))}]'
    return _format(val)


def _format(val) -> str:
    """Return ``<class uid>`` for an object, or ``<class id>`` if it has no uid."""
    if isinstance(val, gws.Root):
        return '<root>'
    if isinstance(val, gws.Application):
        return '<app>'
    uid = vars(val).get('uid') if _is_object(val) else None
    ident = uid if isinstance(uid, str) and uid else hex(id(val))
    return f'<{_class_name(val)} {ident}>'


def _class_name(val) -> str:
    cls = type(val)
    return cls.__module__ + '.' + cls.__qualname__


def _base_type(val) -> str:
    for t in (bool, int, float, str):
        if isinstance(val, t):
            return t.__name__
    return 'NoneType'


def _primitive_str(val) -> str:
    if val is None:
        return 'None'
    if isinstance(val, str):
        return str.__str__(val)
    if isinstance(val, bool):
        return 'True' if val else 'False'
    if isinstance(val, int):
        return int.__repr__(val)
    return float.__repr__(val)


def _repr(val) -> str:
    try:
        s = repr(val)
    except Exception as exc:
        s = f'<repr error: {exc}>'
    if len(s) > _MAX_REPR_LENGTH:
        s = s[:_MAX_REPR_LENGTH] + '...'
    return s


def _is_primitive(val) -> bool:
    return val is None or isinstance(val, (bool, int, float, str))


def _is_list(val) -> bool:
    return isinstance(val, (list, tuple, set, frozenset))


def _is_object(val) -> bool:
    """Check if a value is a dict or an object with attributes that can be inspected."""
    if isinstance(val, dict):
        return True
    if isinstance(val, _OPAQUE_TYPES):
        return False
    return hasattr(val, '__dict__')


def _is_tree_node(root: gws.Root, val) -> bool:
    """Check if a value is a node registered in the root by its uid."""
    return isinstance(val, gws.Node) and root.uidMap.get(vars(val).get('uid')) is val
