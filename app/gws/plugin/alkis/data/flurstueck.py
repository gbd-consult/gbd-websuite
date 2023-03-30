"""ALKIS Flurstuecke -- indexer"""

import collections
import re

import gws
import gws.types as t

from . import resolver, adresse, nutzung, grundbuch
from ..util import indexer
from ..util.connection import AlkisConnection
from gws.lib.console import ProgressIndicator

main_index = 'idx_flurstueck'
name_index = 'idx_name'


class _Cache:
    addr: dict[str, list[dict]] = {}
    gebaeude: dict[str, list[dict]] = {}
    gemarkung: dict[str, str] = {}
    name: list[dict] = []
    nutzung: dict[str, list[dict]] = {}
    buchungsstelle: dict[str, dict] = {}


# vollnummer = flur-zaeher/nenner (folge)

def _vollnummer(fs):
    v = ''
    s = fs.get('flurnummer')
    if s:
        v = str(s) + '-'

    v += str(fs['zaehler'])
    s = fs.get('nenner')
    if s:
        v += '/' + str(s)

    s = fs.get('flurstuecksfolge')
    if s and str(s) != '00':
        v += ' (' + str(s) + ')'

    return v


# parse a vollnummer in the above format, all parts are optional

_vnum_re = r'''^(?x)
    (
        (?P<flurnummer> [0-9]+)
        -
    )?
    (
        (?P<zaehler> [0-9]+)
        (/
            (?P<nenner> \w+)
        )?
    )?
    (
        \(
            (?P<flurstuecksfolge> [0-9]+)
        \)
    )?
    $
'''


def _parse_vollnummer_element(key, val):
    if key == 'flurnummer' or key == 'zaehler':
        try:
            return int(val)
        except ValueError:
            return None

    if key == 'flurstuecksfolge':
        try:
            return '%02d' % int(val)
        except ValueError:
            return None

    return val


def _parse_vollnummer(s):
    s = re.sub(r'\s+', '', s)

    # search by gml_id
    if s.startswith('DE'):
        return {'gml_id': s}

    m = re.match(_vnum_re, s)
    if not m:
        return None

    d = {}
    for k, v in m.groupdict().items():
        v = _parse_vollnummer_element(k, v)
        if v is None:
            return None
        d[k] = v
    return d


def _add_vnum_param(key, val, where, parms):
    if val:
        where.append(f'FS.' + key + ' = %s')
        if key in ('flurnummer', 'zaehler'):
            parms.append(int(val))
        elif key == 'flurstuecksfolge':
            parms.append()
        else:
            parms.append(val)


def _parse_vnum(s, where, parms):
    s = re.sub(r'\s+', '', s)

    # search by gml_id

    if s.startswith('DE'):
        _add_vnum_param('gml_id', s, where, parms)
        return True

    # search by flur-zahler/nenner

    m = re.match(_vnum_re, s)
    if not m:
        return False

    # flurnummer|integer
    # zaehler|integer
    # nenner|character varying
    # flurstuecksfolge|character varying

    for k, v in m.groupdict().items():
        _add_vnum_param(k, v, where, parms)

    return True


def _cache(conn: AlkisConnection):
    idx = conn.index_schema

    cache = _Cache()

    gws.log.info('fs index: nutzung cache')

    nu_parts: dict[str, dict] = {}
    rs = conn.select(f'SELECT * FROM {idx}.{nutzung.parts_index}')

    for r in rs:
        fs_id = r['fs_id']

        if fs_id not in nu_parts:
            nu_parts[fs_id] = {}

        nu = {
            'type': r['type'],
            'type_id': r['type_id'],
            'area': r['area'],
            'a_area': r['a_area'],
            'gml_id': r['nu_id'],
            'count': 1
        }

        if r['attributes']:
            nu.update(indexer.from_json(r['attributes']))

        k = (nu['type_id'], nu['key_id'])

        if k not in nu_parts[fs_id]:
            nu_parts[fs_id][k] = nu
        else:
            nu_parts[fs_id][k]['area'] += nu['area']
            nu_parts[fs_id][k]['a_area'] += nu['a_area']
            nu_parts[fs_id][k]['count'] += 1

    cache.nutzung = {}

    for fs_id, nu_dict in nu_parts.items():
        cache.nutzung[fs_id] = sorted(nu_dict.values(), key=lambda nu: -nu['area'])

    gws.log.info('fs index: grundbuch cache')

    bs = {}
    rs = conn.select(f'SELECT * FROM {idx}.{grundbuch.stelle_index}')
    for r in rs:
        bs[r['gml_id']] = indexer.from_json(r['stellen'])

    cache.buchungsstelle = bs

    gws.log.info('fs index: adresse cache')

    addr = collections.defaultdict(list)
    rs = conn.select(f'SELECT * FROM {idx}.{adresse.addr_index}')

    for r in rs:
        fs_id = r.pop('fs_id')
        addr[fs_id].append(r)

    cache.addr = dict(addr)

    cache.gemarkung = {}

    rs = conn.select(f'SELECT DISTINCT gemarkung_id, gemarkung_v FROM {idx}.{adresse.addr_index}')
    for r in rs:
        cache.gemarkung[r['gemarkung_id']] = r['gemarkung_v']

    gws.log.info('fs index: gebaeude cache')

    geb = collections.defaultdict(list)
    rs = conn.select(f'SELECT gml_id, fs_id, attributes, area FROM {idx}.{adresse.gebs_index}')

    for r in rs:
        if r['attributes']:
            r.update(indexer.from_json(r.pop('attributes')))
        fs_id = r.pop('fs_id')
        geb[fs_id].append(r)

    cache.gebaeude = dict(geb)

    cache.name = []

    return cache


def _fs_data(conn, fs, cache):
    keys = [
        'gml_id',
        'flurnummer',
        'zaehler',
        'nenner',
        'flurstueckskennzeichen',
        'flurstuecksfolge',
        'gemarkungsnummer',
        'amtlicheflaeche',
        'zeitpunktderentstehung',

        'geom',
        'x',
        'y',
    ]

    d = {k: fs.get(k, '') for k in keys}

    fs_id = fs['gml_id']

    d['zaehlernenner'] = str(d['zaehler'])
    if d['nenner']:
        d['zaehlernenner'] += '/' + str(d['nenner'])

    d['vollnummer'] = _vollnummer(d)

    d.update(resolver.places(conn, fs))
    d['gemarkung_v'] = cache.gemarkung.get(d.get('gemarkung_id')) or d.get('gemarkung')
    d.pop('gemarkungsnummer', None)

    p = cache.addr.get(fs_id, [])
    p.sort(key=lambda x: x.get('strasse_v', ''))
    d['lage'] = indexer.to_json(p)
    d['c_lage'] = len(p)

    p = cache.gebaeude.get(fs_id, [])
    p.sort(key=lambda x: -x.get('area'))
    d['gebaeude'] = indexer.to_json(p)
    d['c_gebaeude'] = len(p)
    d['gebaeude_area'] = sum(x['area'] for x in p)

    stellen = cache.buchungsstelle.get(fs['istgebucht'], [])
    bb_number = []

    for stelle in stellen:
        if 'buchungsblatt' in stelle:
            bbn = stelle['buchungsblatt'].get('buchungsblattkennzeichen')
            if bbn:
                bb_number.append(bbn)
        if 'eigentuemer' in stelle:
            for eigentuemer in stelle['eigentuemer']:
                if 'person' in eigentuemer:
                    cache.name.append({
                        'fs_id': fs_id,
                        'nachname': eigentuemer['person'].get('nachnameoderfirma'),
                        'vorname': eigentuemer['person'].get('vorname'),
                    })

    d['c_buchung'] = len(stellen)
    d['buchung'] = indexer.to_json(stellen)

    # bb_number list must be like ';nnn;nnn;nnn;' (see search below)
    d['bb_number'] = ''
    if bb_number:
        d['bb_number'] = ';' + ';'.join(s.strip() for s in bb_number)

    p = cache.nutzung.get(fs_id, [])
    d['nutzung'] = indexer.to_json(p)
    d['c_nutzung'] = len(p)

    return d


def _create_main_index(conn: AlkisConnection):
    idx = conn.index_schema
    dat = conn.data_schema

    cache = _cache(conn)

    cond = {
        'endet': '?? IS NULL'
    }

    if conn.exclude_gemarkung:
        # exclude a list of "gemarkungsnummer" (string)
        excl = ','.join("'%d'" % int(x) for x in conn.exclude_gemarkung)
        cond['gemarkungsnummer'] = f'?? NOT IN ({excl})'

    rs = conn.select_from_ax('ax_flurstueck', [
        'gml_id',
        'flurnummer',
        'zaehler',
        'nenner',
        'flurstueckskennzeichen',
        'flurstuecksfolge',
        'amtlicheflaeche',
        'zeitpunktderentstehung',
        'istgebucht',

        'land',
        'kreis',
        'regierungsbezirk',
        'gemeinde',
        'gemarkungsnummer',

        'wkb_geometry AS geom',
        'ST_X(ST_Centroid(wkb_geometry)) AS x',
        'ST_Y(ST_Centroid(wkb_geometry)) AS y'
    ], conditions=cond)

    data = []
    total = conn.count(f'{dat}.ax_flurstueck')

    with ProgressIndicator('flurstueck: reading', total) as pi:
        for fs in rs:
            data.append(_fs_data(conn, fs, cache))
            pi.update(1)

    gws.log.info('flurstueck: writing')

    conn.create_index_table(main_index, f'''
        gml_id CHARACTER(16) NOT NULL PRIMARY KEY,

        land CHARACTER VARYING,
        land_id CHARACTER VARYING,
        regierungsbezirk CHARACTER VARYING,
        regierungsbezirk_id CHARACTER VARYING,
        kreis  CHARACTER VARYING,
        kreis_id CHARACTER VARYING,
        gemeinde CHARACTER VARYING,
        gemeinde_id CHARACTER VARYING,
        gemarkung  CHARACTER VARYING,
        gemarkung_id CHARACTER VARYING,
        gemarkung_v  CHARACTER VARYING,

        anlass CHARACTER VARYING,
        endet CHARACTER VARYING,
        zeitpunktderentstehung CHARACTER VARYING,
        amtlicheflaeche FLOAT,
        flurnummer INTEGER,
        zaehler INTEGER,
        nenner CHARACTER VARYING,
        flurstuecksfolge CHARACTER VARYING,

        zaehlernenner CHARACTER VARYING,
        vollnummer CHARACTER VARYING,

        flurstueckskennzeichen CHARACTER(20),

        lage CHARACTER VARYING,
        c_lage INTEGER,

        gebaeude CHARACTER VARYING,
        gebaeude_area FLOAT,
        c_gebaeude INTEGER,

        buchung CHARACTER VARYING,
        c_buchung INTEGER,

        bb_number CHARACTER VARYING,

        nutzung CHARACTER VARYING,
        c_nutzung INTEGER,

        geom geometry(GEOMETRY, {conn.srid}),
        x FLOAT,
        y FLOAT
    ''')
    conn.index_insert(main_index, data)

    gws.log.info('flurstueck: writing names')

    conn.create_index_table(name_index, f'''
        fs_id CHARACTER(16) NOT NULL,
        vorname CHARACTER VARYING,
        nachname CHARACTER VARYING
    ''')
    conn.index_insert(name_index, cache.name)

    conn.mark_index_table(main_index)
    conn.mark_index_table(name_index)

    gws.log.info('flurstueck: done')


def create_index(conn):
    if not indexer.check_version(conn, main_index):
        _create_main_index(conn)


def index_ok(conn):
    return indexer.check_version(conn, main_index)


def gemarkung_list(conn):
    return conn.select(f'''
        SELECT DISTINCT 
            gemarkung_id AS "gemarkungUid", 
            gemarkung, 
            gemeinde_id AS "gemeindeUid", 
            gemeinde
        FROM {conn.index_schema}.{main_index}
        WHERE gemarkung_id IS NOT NULL
    ''')


def _strasse_condition(alias, query) -> t.Optional[gws.Sql]:
    if 'strasse' not in query:
        return None

    skey = adresse.street_name_key(query['strasse'])
    smode = query.get('strasseMode', 'exact')

    if smode == 'exact':
        return gws.Sql(alias + '.strasse_k = {}', skey)

    if smode == 'start':
        return gws.Sql(alias + '.strasse_k = {:like}', ['*%', skey])

    if smode == 'substring':
        return gws.Sql(alias + '.strasse_k = {:like}', ['%*%', skey])


def strasse_list(conn: AlkisConnection, query: dict):
    where = [
        gws.Sql("strasse NOT IN ('ohne Lage')")
    ]

    if conn.exclude_gemarkung:
        where.append(gws.Sql('gemarkung_id NOT IN ({:values})', conn.exclude_gemarkung))

    for key, val in query.items():
        if gws.is_empty(val):
            continue

        if key == 'gemeindeUid':
            where.append(gws.Sql('gemeinde_id={}', val))
        elif key == 'gemarkungUid':
            where.append(gws.Sql('gemarkung_id={}', val))

    sk = _strasse_condition('AD', query)
    if sk:
        where.append(sk)

    return conn.select('''
        SELECT DISTINCT
            strasse,
            gemeinde_id as "gemeindeUid",
            gemeinde,
            gemarkung_id as "gemarkungUid",
            gemarkung
        FROM {:qname} AS AD
        WHERE {:and}
        ORDER BY strasse
    ''', [conn.index_schema, adresse.addr_index], where)


def has_flurnummer(conn: AlkisConnection):
    n = conn.select_value(f'''
        SELECT COUNT(*)
        FROM {conn.index_schema}.{main_index}
        WHERE flurnummer IS NOT NULL
    ''')
    return n > 0


def find(conn: AlkisConnection, query: dict):
    where = []

    tables = {
        'FS': main_index
    }

    for key, val in query.items():
        if gws.is_empty(val):
            continue

        if key == 'gemarkungUid':
            where.append(gws.Sql('FS.gemarkung_id = {}', val))

        elif key == 'gemeindeUid':
            where.append(gws.Sql('FS.gemeinde_id = {}', val))

        elif key == 'strasse':
            tables['AD'] = adresse.addr_index
            where.append(gws.Sql('AD.fs_id = FS.gml_id'))

            sk = _strasse_condition('AD', query)
            if sk:
                where.append(sk)

            hnr = query.get('hausnummer')
            if hnr == '*':
                where.append(gws.Sql('AD.hausnummer IS NOT NULL'))
            elif hnr:
                where.append(gws.Sql('AD.hausnummer = {}', adresse.normalize_hausnummer(hnr)))

        elif key == 'name':
            tables['NA'] = name_index
            where.append(gws.Sql('NA.fs_id = FS.gml_id'))
            where.append(gws.Sql('NA.nachname ILIKE {:like}', ['*%', val]))

            v = query.get('vorname')
            if v:
                where.append(gws.Sql('NA.vorname ILIKE {:like}', ['*%', v]))

        elif key == 'vorname':
            if not query.get('name'):
                return 0, []

        elif key == 'flaecheVon':
            where.append(gws.Sql('FS.amtlicheflaeche >= {}', val))

        elif key == 'flaecheBis':
            where.append(gws.Sql('FS.amtlicheflaeche <= {}', val))

        elif key == 'bblatt':
            # bblatt numbers are ';nnn;nnn;nnn;', see above
            # the input can be ; or , or space separated

            nums = []
            for s in val.replace(';', ' ').replace(',', ' ').strip().split():
                if not s.isdigit():
                    gws.log.warn('invalid bblatt', val)
                    return 0, []
                nums.append(s)

            if not nums:
                gws.log.warn('invalid bblatt', val)
                return 0, []

            bbmode = query.get('bblattMode', 'any')
            for n in nums:
                if bbmode == 'exact':
                    where.append(gws.Sql('FS.bb_number LIKE {}', f'%;{n};%'))
                if bbmode == 'start':
                    where.append(gws.Sql('FS.bb_number LIKE {}', f'%;{n}%'))
                if bbmode == 'end':
                    where.append(gws.Sql('FS.bb_number LIKE {}', f'%{n};%'))
                if bbmode == 'any':
                    where.append(gws.Sql('FS.bb_number LIKE {}', f'%{n}%'))

        elif key == 'shape':
            where.append(gws.Sql('ST_Intersects({}::geometry, FS.geom)', val.transformed_to(conn.crs).ewkb_hex))

        elif key == 'vnum':
            d = _parse_vollnummer(val)
            if not d:
                gws.log.warn('invalid vnum', val)
                return 0, []
            for k, v in d:
                where.append(gws.Sql('FS.{:name}={}', k, v))

        elif key in {'flurnummer', 'zaehler', 'nenner', 'flurstuecksfolge'}:
            v = _parse_vollnummer_element(key, val)
            if v is None:
                gws.log.warn(f'invalid element {key!r}', val)
                return 0, []
            where.append(gws.Sql('FS.{:name}={}', key, v))

        elif key == 'fsUids':
            where.append(gws.Sql('FS.gml_id IN ({:values})', val))

    from_part = gws.Sql(','.join(f'{conn.index_schema}.{v} AS {k}' for k, v in tables.items()))

    count = conn.select_value(
        'SELECT COUNT(DISTINCT FS.*) FROM {:sql} {:sql}',
        from_part,
        gws.Sql('WHERE {:and}', where) if where else None
    )

    limit = query.get('limit')

    data = conn.select(
        'SELECT DISTINCT FS.* FROM {:sql} {:sql} {:sql}',
        from_part,
        gws.Sql('WHERE {:and}', where) if where else None,
        gws.Sql('LIMIT {:int}', limit) if limit else None
    )

    for fs in data:
        for k in 'lage', 'buchung', 'nutzung', 'gebaeude':
            v = indexer.from_json(fs.get(k))
            if v:
                fs[k] = v

    return count, data
