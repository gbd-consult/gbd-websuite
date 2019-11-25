import re
import itertools

import gws
import gws.common.csv

"""
A fs structure, as created by our indexer, is deeply nested.
We flatten it first, creating a list 'some.nested.key, list positions, value'
    
    ...then filter out unwanted keys
    
    ...then create a product of all list positions, so if there are 3 'lage' lists
    and 2 'eigentuemer' lists, there will be 3x2=6 rows
"""

# actual fs 'flat' keys, classified by their usage ('part')


_groups = {
    'base': [
        'gml_id',
        'land',
        'regierungsbezirk',
        'kreis',
        'gemeinde',
        'gemarkung_id',
        'gemarkung',
        'anlass',
        'endet',
        'zeitpunktderentstehung',
        'flurnummer',
        'zaehler',
        'nenner',
        'flurstuecksfolge',
        'amtlicheflaeche',
        'zaehlernenner',
        'vollnummer',
        'flurstueckskennzeichen',
        'x',
        'y',

    ],
    'lage': [
        'lage.gemarkung',
        'lage.gemarkung_id',
        'lage.gemeinde',
        'lage.gemeinde_id',
        'lage.gml_id',
        'lage.kreis',
        'lage.kreis_id',
        'lage.lage_id',
        'lage.lage_schluesselgesamt',
        'lage.land',
        'lage.land_id',
        'lage.regierungsbezirk',
        'lage.regierungsbezirk_id',
        'lage.strasse',
        'lage.hausnummer',
        'lage.x',
        'lage.xysrc',
        'lage.y',
    ],
    'gebaeude': [
        'gebaeude.area',
        'gebaeude.gebaeudefunktion',
        'gebaeude.gebaeudefunktion_id',
        'gebaeude.gml_id',
    ],
    'buchung': [
        'buchung.beginnt',
        'buchung.buchungsart',
        'buchung.buchungsart_id',
        'buchung.buchungsblatt.bezirk',
        'buchung.buchungsblatt.bezirk_id',
        'buchung.buchungsblatt.blattart',
        'buchung.buchungsblatt.blattart_id',
        'buchung.buchungsblatt.buchungsblattkennzeichen',
        'buchung.buchungsblatt.buchungsblattnummermitbuchstabenerweiterung',
        'buchung.buchungsblatt.gml_id',
        'buchung.buchungsblatt.land',
        'buchung.buchungsblatt.land_id',
        'buchung.gml_id',
        'buchung.laufendenummer',
    ],

    'eigentuemer': [
        'buchung.eigentuemer.gml_id',
        'buchung.eigentuemer.anteil',
        'buchung.eigentuemer.eigentuemerart',
        'buchung.eigentuemer.eigentuemerart_id',
        'buchung.eigentuemer.laufendenummernachdin1421',

        'buchung.eigentuemer.person.gml_id',
        'buchung.eigentuemer.person.akademischergrad',
        'buchung.eigentuemer.person.anrede',
        'buchung.eigentuemer.person.vorname',
        'buchung.eigentuemer.person.nachnameoderfirma',
        'buchung.eigentuemer.person.geburtsdatum',

        'buchung.eigentuemer.person.anschrift.gml_id',
        'buchung.eigentuemer.person.anschrift.hausnummer',
        'buchung.eigentuemer.person.anschrift.ort_post',
        'buchung.eigentuemer.person.anschrift.postleitzahlpostzustellung',
        'buchung.eigentuemer.person.anschrift.strasse',
    ],

    'nutzung': [
        'nutzung.a_area',
        'nutzung.area',
        'nutzung.count',
        'nutzung.gml_id',
        'nutzung.key',
        'nutzung.key_id',
        'nutzung.key_label',
        'nutzung.type',
        'nutzung.type_id',
    ]
}

_headers = {
    'gemeinde': 'Gemeinde',
    'gemarkung_id': 'Gemarkung ID',
    'gemarkung': 'Gemarkung',
    'amtlicheflaeche': 'Fläche',
    'flurnummer': 'Flurnummer',
    'zaehler': 'Zähler',
    'nenner': 'Nenner',
    'flurstuecksfolge': 'Folge',
    'x': 'X',
    'y': 'Y',

    'lage.strasse': 'FS Strasse',
    'lage.hausnummer': 'FS Hnr',

    'gebaeude.area': 'Gebäude Fläche',
    'gebaeude.gebaeudefunktion': 'Gebäude Funktion',

    'buchung.buchungsart': 'Buchungsart',
    'buchung.laufendenummer': 'Laufende Nummer',
    'buchung.buchungsblatt.blattart': 'Blattart',
    'buchung.buchungsblatt.buchungsblattkennzeichen': 'Blattkennzeichen',
    'buchung.buchungsblatt.buchungsblattnummermitbuchstabenerweiterung': 'Blattnummer',

    'buchung.eigentuemer.person.vorname': 'Vorname',
    'buchung.eigentuemer.person.nachnameoderfirma': 'Name',
    'buchung.eigentuemer.person.geburtsdatum': 'Geburtsdatum',
    'buchung.eigentuemer.person.anschrift.strasse': 'Strasse',
    'buchung.eigentuemer.person.anschrift.hausnummer': 'Hnr',
    'buchung.eigentuemer.person.anschrift.postleitzahlpostzustellung': 'PLZ',
    'buchung.eigentuemer.person.anschrift.ort_post': 'Ort',

    'nutzung.a_area': 'Nutzung Fläche',
    'nutzung.type': 'Nutzung Typ',
}


def _format_value(val, rec):
    def repl(m):
        ref = m.group(1)
        if ref.endswith('[0]'):
            v = rec.get(ref[:-3]) or ''
            return v[:1]
        return rec.get(ref, '')

    if '{' not in val:
        return val

    return re.sub(r'{(.+?)}', repl, val)

def as_csv(obj, fs_list, groups, path, headers=None):
    keys = []
    for g in groups:
        keys.extend(_groups[g])

    rows = []

    for fs in fs_list:
        fs_as_csv(fs, keys, rows)

    if headers:
        headers = [h for h in headers if h['source'] in keys]
        csv_rows = []
        for r in rows:
            r = dict(zip(keys, r))
            rf = []
            for h in headers:
                if 'value' in h:
                    rf.append(_format_value(h['value'], r))
                else:
                    rf.append(r[h['source']])
            csv_rows.append(rf)
        csv_headers = [h['title'] for h in headers]
    else:
        use_keys = [k for k in _headers if k in keys]
        csv_headers = [_headers[k] for k in use_keys]
        csv_rows = []
        for r in rows:
            r = dict(zip(keys, r))
            csv_rows.append([r.get(k) for k in use_keys])

    csv: gws.common.csv.Object = obj.root.find_first('gws.common.csv')
    csv.write(path, csv_headers, csv_rows)


def fs_as_csv(fs, keys, rows):
    flat = _flatten(fs)

    # keep keys we need

    flat = [
        f
        for f in flat
        if any(f[0].startswith(k) for k in keys)
    ]

    # collect lists

    lst = _get_lists(flat)

    # no lists, write a single row

    if not lst:
        _write_flat(flat, keys, rows)
        return

    # create a row for each combination of list indexes

    lst_ranges = [range(x + 1) for x in lst.values()]

    for lst_indexes in itertools.product(*lst_ranges):
        matching = [
            f
            for f in flat
            if _indexes_match(f, lst.keys(), lst_indexes)
        ]
        _write_flat(matching, keys, rows)


def _flatten(fs):
    flat = []

    """
    create a flat list from a nested fs record
    an element of the list is (path, pos, value)
    where 
        path = full key path, dot separated
        pos  = {list_name:index, ...} if a value is in an list
        value
    """

    _flat_walk(flat, fs, '', {})
    return flat


def _flat_walk(flat, obj, path, pos):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flat_walk(flat, v, path + '.' + str(k) if path else str(k), pos)

    elif isinstance(obj, list):
        if obj:
            for n, v in enumerate(obj):
                p = dict(pos)
                p[path] = n
                _flat_walk(flat, v, path, p)

    else:
        flat.append((path, pos, obj))


def _get_lists(flat):
    lst = {}

    """
    extract lists from pos elements of the flat list
    return a dict {list_name => max_index}
    """

    for path, pos, val in flat:
        for k, v in pos.items():
            lst[k] = max(lst.get(k, 0), v)

    return lst


def _indexes_match(flat_elem, lst_keys, lst_indexes):
    pos = flat_elem[1]

    """
    check if a flat entry matches the given combination
    of list positions
    """

    for k, i in zip(lst_keys, lst_indexes):
        p = pos.get(k)
        if p is not None and p != i:
            return False

    return True


def _write_flat(flat, keys, rows):
    r = {}
    for path, pos, val in flat:
        r[path] = val
    rows.append([r.get(k) for k in keys])
