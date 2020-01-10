"""Interface for Objektartengruppe:Personen- und Bestandsdaten"""

import gws
from . import resolver
from ..util import indexer, connection

stelle_index = 'idx_grundbuch_stelle'

"""
ALKIS relations:

ax_flurstueck     --- [istGebucht]        ---> ax_buchungsstelle
ax_buchungsstelle --- [istBestandteilVon] ---> ax_buchungsblatt
ax_namensnummer   --- [istBestandteilVon] ---> ax_buchungsblatt
ax_namensnummer   --- [benennt]           ---> ax_person
ax_person         --- [hat]               ---> ax_anschrift

"""


def _anteil(r):
    if r['nenner'] is None:
        return None
    return '%d/%d' % (r['zaehler'] or 0, r['nenner'] or 0)


def _all_person(conn):
    dat = conn.data_schema

    rs = conn.select_from_ax('ax_anschrift', [
        'gml_id',
        'ort_post',
        'ortsteil',
        'postleitzahlpostzustellung',
        'strasse',
        'hausnummer',
        'telefon'
    ])

    anschrift = {r['gml_id']: gws.compact(r) for r in rs}
    person = {}

    anrede = {
        1000: 'Frau',
        2000: 'Herr',
        3000: 'Firma'
    }

    rs = conn.select_from_ax('ax_person', [
        'gml_id',
        'anrede',
        'akademischergrad',
        'geburtsdatum',
        'nachnameoderfirma',
        'vorname',
        'hat'
    ])

    for r in rs:
        hat = r.pop('hat', None) or []
        for h in hat:
            if h in anschrift:
                r['anschrift'] = anschrift[h]
                break
        r['anrede'] = anrede.get(r['anrede'])
        person[r['gml_id']] = gws.compact(r)

    return person


def _all_buchungsblatt(conn):
    persons = _all_person(conn)
    blatts = {}

    rs = conn.select_from_ax('ax_buchungsblatt')

    for r in rs:
        bb = {
            'gml_id': r['gml_id'],
            'buchungsblattkennzeichen': r['buchungsblattkennzeichen'],
            'buchungsblattnummermitbuchstabenerweiterung': r['buchungsblattnummermitbuchstabenerweiterung'],
            'eigentuemer': [],
        }
        bb.update(resolver.places(conn, r))
        bb.update(resolver.attributes(conn, 'ax_buchungsblatt', r))
        blatts[r['gml_id']] = gws.compact(bb)

    rs = conn.select_from_ax('ax_namensnummer')

    for r in rs:
        if r['istbestandteilvon'] in blatts:
            eigentuemer = {
                'anteil': _anteil(r),
                'gml_id': r['gml_id'],
                'laufendenummernachdin1421': r['laufendenummernachdin1421'],
                'person': persons.get(r['benennt'])
            }
            eigentuemer.update(resolver.attributes(conn, 'ax_namensnummer', r))
            blatts[r['istbestandteilvon']]['eigentuemer'].append(gws.compact(eigentuemer))

    return blatts


# for each 'buchungsstelle.gml_id', create a list of 'buchungsstelle'
# + its 'an' dependants, recursively

def _make_list(stellen, stelle, seen_ids):
    if stelle['gml_id'] in seen_ids:
        gws.log.warn('circular dependency: ' + stelle['gml_id'])
        return []

    slist = [stelle]
    seen_ids.add(stelle['gml_id'])

    for gml_id in stelle.get('_an', []):
        slist.extend(_make_list(stellen, stellen[gml_id], seen_ids))

    return slist


def _all_buchungsstelle(conn):
    blatts = _all_buchungsblatt(conn)
    stellen = {}

    rs = conn.select_from_ax('ax_buchungsstelle')

    for r in rs:
        bb = blatts.get(r['istbestandteilvon'], {})
        st = {
            'gml_id': r['gml_id'],
            'beginnt': r['beginnt'],
            'endet': r['endet'],
            'laufendenummer': r['laufendenummer'],
            'an': r['an'],
            'eigentuemer': bb.get('eigentuemer', []),
            'buchungsblatt': {k: v for k, v in bb.items() if k != 'eigentuemer'},
            'anteil': _anteil(r),
        }
        st.update(resolver.places(conn, r))
        st.update(resolver.attributes(conn, 'ax_buchungsstelle', r))
        stellen[r['gml_id']] = gws.compact(st)

    # resolve 'an' dependencies
    # Eine 'Buchungsstelle' verweist mit 'an' auf eine andere 'Buchungsstelle' auf einem anderen Buchungsblatt.
    # Die Buchungsstelle kann ein Recht (z.B. Erbbaurecht) oder einen Miteigentumsanteil 'an' der anderen Buchungsstelle haben
    # Die Relation zeigt stets vom begünstigten Recht zur belasteten Buchung
    # (z.B. Erbbaurecht hat ein Recht 'an' einem Grundstück).

    for r in stellen.values():
        for gml_id in r.get('an', []):
            if gml_id is None:
                continue
            if gml_id not in stellen:
                gws.log.warn('invalid "an" reference: ' + str(id))
                continue
            target = stellen[gml_id]
            if '_an' not in target:
                target['_an'] = []
            target['_an'].append(r['gml_id'])

    data = []
    for r in stellen.values():
        ls = _make_list(stellen, r, set())
        data.append({
            'gml_id': r['gml_id'],
            'stellen': indexer.as_json(ls)
        })

    return data


def _create_stelle_index(conn: connection.AlkisConnection):
    data = _all_buchungsstelle(conn)

    conn.create_index_table(stelle_index, f'''
            id SERIAL PRIMARY KEY,
            gml_id CHARACTER VARYING,
            stellen CHARACTER VARYING
    ''')
    conn.index_insert(stelle_index, data)
    conn.mark_index_table(stelle_index)


def create_index(conn):
    if not indexer.check_version(conn, stelle_index):
        _create_stelle_index(conn)


def index_ok(conn):
    return indexer.check_version(conn, stelle_index)
