import itertools

import gws
import gws.base.csv
import gws.base.model
import gws.types as t

"""
A fs structure, as created by our indexer, is deeply nested.
We flatten it first, creating a list 'some.nested.key, list positions, value'

    ...then filter out unwanted keys

    ...then create a product of all list positions, so if there are 3 'lage' lists
    and 2 'eigentuemer' lists, there will be 3x2=6 rows
"""


def to_csv(action: gws.INode, fs_features: t.List[gws.IFeature], model: gws.base.model.Object):
    helper: gws.base.csv.Object = action.root.application.require_helper('csv')

    writer = helper.writer()
    writer.write_headers([r.title for r in model.rules])

    for fs in fs_features:
        for rec in _recs_from_feature(fs, model.attribute_names):
            writer.write_attributes(model.apply_to_dict(rec))

    return writer.to_bytes()


def _recs_from_feature(fs: gws.IFeature, att_names: t.List[str]):
    # create a flat list from the attributes of the FS feature

    flat = list(_flat_walk({a.name: a.value for a in fs.attributes}))

    # keep keys we need

    flat = [e for e in flat if any(e['path'].startswith(a) for a in att_names)]

    # compute max index for each list from 'pos' elements

    max_index: t.Dict[str, int] = {}

    for e in flat:
        for k, v in e['pos'].items():
            max_index[k] = max(max_index.get(k, 0), v)

    # no lists, return a single record

    if not max_index:
        yield {e['path']: e['value'] for e in flat}
        return

    # create a record for each combination of list indexes

    list_keys = max_index.keys()
    list_ranges = [range(x + 1) for x in max_index.values()]

    for list_indexes in itertools.product(*list_ranges):
        matching = [e for e in flat if _indexes_match(e, list_keys, list_indexes)]
        yield {e['path']: e['value'] for e in matching}


def _flat_walk(obj, path=None, pos=None):
    # create a flat list from a nested fs record
    # an element of the list is {path, pos, value}, where
    #     path = full key path (joined by _)
    #     pos  = {list_name: list_index, ...} if a value is a member of a list
    #     value = element value

    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat_walk(v, path + '_' + str(k) if path else str(k), pos)
        return

    if isinstance(obj, list):
        for n, v in enumerate(obj):
            p = dict(pos) if pos else {}
            p[path] = n
            yield from _flat_walk(v, path, p)
        return

    yield {'path': path, 'pos': pos or {}, 'value': obj}


def _indexes_match(flat_elem, list_keys, list_indexes):
    # check if a flat entry matches the given combination of list positions

    for k, i in zip(list_keys, list_indexes):
        p = flat_elem['pos'].get(k)
        if p is not None and p != i:
            return False

    return True
