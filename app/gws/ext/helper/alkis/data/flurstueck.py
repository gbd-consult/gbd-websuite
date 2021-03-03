"""ALKIS Flurstuecke -- indexer"""

import collections
import re

import gws
from gws.tools.console import ProgressIndicator

from . import resolver, adresse, nutzung, grundbuch
from ..util import indexer
from ..util.connection import AlkisConnection

main_index = 'idx_flurstueck'
name_index = 'idx_name'


class _Cache:
    addr = {}
    gebaeude = {}
    gemarkung = {}
    name = {}
    nutzung = {}
    buchungsstelle = {}


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


def _add_vnum_param(key, val, where, parms):
    if val:
        where.append(f'FS.' + key + ' = %s')
        if key in ('flurnummer', 'zaehler'):
            parms.append(int(val))
        elif key == 'flurstuecksfolge':
            parms.append('%02d' % int(val))
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

    nu_parts = {}
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

    for fs_id in nu_parts:
        nu_parts[fs_id] = sorted(nu_parts[fs_id].values(), key=lambda x: -x['area'])

    cache.nutzung = nu_parts

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

    p = cache.addr.get(fs_id, [])
    p.sort(key=lambda x: x.get('strasse_v', ''))
    d['lage'] = indexer.as_json(p)
    d['c_lage'] = len(p)

    p = cache.gebaeude.get(fs_id, [])
    p.sort(key=lambda x: -x.get('area'))
    d['gebaeude'] = indexer.as_json(p)
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
    d['buchung'] = indexer.as_json(stellen)

    d['bb_number'] = ';'.join(bb_number) + ';' if bb_number else ''

    p = cache.nutzung.get(fs_id, [])
    d['nutzung'] = indexer.as_json(p)
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
        'gemarkungsnummer',
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
        regierungsbezirk CHARACTER VARYING,
        kreis CHARACTER VARYING,
        
        gemeinde_id CHARACTER VARYING,
        gemeinde CHARACTER VARYING,
        
        gemarkung_id CHARACTER VARYING,
        gemarkung CHARACTER VARYING,
        gemarkung_v CHARACTER VARYING,
        
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
    rs = conn.select(f'''
        SELECT DISTINCT gemarkung_id as "gemarkungUid", gemarkung, gemeinde_id as "gemeindeUid", gemeinde
        FROM {conn.index_schema}.{main_index}
        WHERE gemarkung_id IS NOT NULL
    ''')
    return list(rs)


def _add_strasse_param(query, where, parms):
    if 'strasse' not in query:
        return

    skey = adresse.street_name_key(query['strasse'])
    smode = query.get('strasseMode', 'exact')

    if smode == 'exact':
        where.append('AD.strasse_k = %s')
        parms.append(skey)
        return

    if smode == 'start':
        where.append('AD.strasse_k LIKE %s')
        parms.append(skey + '%')
        return

    if smode == 'substring':
        where.append('AD.strasse_k LIKE %s')
        parms.append('%' + skey + '%')
        return


def strasse_list(conn: AlkisConnection, query: dict):
    where = ["strasse NOT IN ('ohne Lage')"]
    parms = []

    if conn.exclude_gemarkung:
        where.append('gemarkung_id NOT IN (%s)' % ','.join(['%s'] * len(conn.exclude_gemarkung)))
        parms.extend(conn.exclude_gemarkung)

    if 'gemeindeUid' in query:
        where.append('gemeinde_id=%s')
        parms.append(query['gemeindeUid'])

    if 'gemarkungUid' in query:
        where.append('gemarkung_id=%s')
        parms.append(query['gemarkungUid'])

    _add_strasse_param(query, where, parms)

    where = ' AND '.join(where)

    rs = conn.select(f'''
        SELECT DISTINCT 
            strasse, 
            gemeinde_id as "gemeindeUid", 
            gemeinde, 
            gemarkung_id as "gemarkungUid",
            gemarkung
        FROM {conn.index_schema}.{adresse.addr_index} as AD
        WHERE {where}
        ORDER BY strasse
    ''', parms)

    return list(rs)


def has_flurnummer(conn: AlkisConnection):
    n = conn.select_value(f'''
        SELECT COUNT(*)
        FROM {conn.index_schema}.{main_index}
        WHERE flurnummer IS NOT NULL
    ''')
    return n > 0


_DEFAULT_LIMIT = 100


def find(conn: AlkisConnection, query: dict):
    where = []
    parms = []

    tables = {
        'FS': main_index
    }

    if not query:
        return 0, []

    if query.get('vorname') and not query.get('name'):
        return 0, []

    def _prepare_for_like(v):
        return v.lower().replace('%', '').replace('_', '') + '%'

    for k, v in query.items():

        if k == 'gemarkungUid':
            where.append('FS.gemarkung_id = %s')
            parms.append(v)

        if k == 'gemeindeUid':
            where.append('FS.gemeinde_id = %s')
            parms.append(v)

        elif k == 'strasse':
            tables['AD'] = adresse.addr_index
            where.append('AD.fs_id = FS.gml_id')

            _add_strasse_param(query, where, parms)

            hnr = query.get('hausnummer')

            if hnr == '*':
                where.append('AD.hausnummer IS NOT NULL')
            elif hnr:
                where.append('AD.hausnummer = %s')
                parms.append(adresse.normalize_hausnummer(hnr))
            # else:
            #     # if no hausnummer, sort by hnr
            #     order.append('AD.hausnummer_n')

        elif k == 'name':
            tables['NA'] = name_index
            where.append('NA.fs_id = FS.gml_id')
            where.append('NA.nachname ILIKE %s')
            parms.append(_prepare_for_like(v))

            if query.get('vorname'):
                where.append('NA.vorname ILIKE %s')
                parms.append(_prepare_for_like(query.get('vorname')))

        elif k == 'flaecheVon':
            where.append('FS.amtlicheflaeche >= %s')
            parms.append(v)

        elif k == 'flaecheBis':
            where.append('FS.amtlicheflaeche <= %s')
            parms.append(v)
            continue

        if k == 'bblatt':
            # bblatt numbers are ; separated, see above
            # @TODO should start with ';' as well
            # the input can be ; or , or space separated

            nums = []
            for s in v.replace(';', ' ').replace(',', ' ').strip().split():
                if not s.isdigit():
                    gws.log.warn('invalid bblatt', v)
                    return 0, []
                nums.append(s)

            if not nums:
                gws.log.warn('invalid bblatt', v)
                return 0, []

            bbmode = query.get('bblattMode', 'any')
            d = ';'
            for n in nums:
                if bbmode == 'exact':
                    where.append(f"('{d}' || FS.bb_number LIKE %s)")
                    parms.append(f'%{d}{n}{d}%')
                if bbmode == 'start':
                    where.append(f"('{d}' || FS.bb_number LIKE %s)")
                    parms.append(f'%{d}{n}%')
                if bbmode == 'end':
                    where.append(f"FS.bb_number LIKE %s")
                    parms.append(f'%{n}{d}%')
                if bbmode == 'any':
                    where.append(f"FS.bb_number LIKE %s")
                    parms.append(f'%{n}%')

        elif k == 'shape':
            v = v.transformed_to(conn.crs)
            where.append(f'ST_Intersects(ST_SetSRID(%s::geometry,%s), FS.geom)')
            parms.append(v.wkb_hex)
            parms.append(conn.srid)

        elif k == 'vnum':
            ok = _parse_vnum(v, where, parms)
            if not ok:
                gws.log.warn('invalid vnum', v)
                return 0, []

        elif k in ('flurnummer', 'zaehler', 'nenner', 'flurstuecksfolge'):
            _add_vnum_param(k, v, where, parms)

        elif k == 'fsUids':
            where.append('FS.gml_id IN (' + ','.join(['%s'] * len(v)) + ')')
            parms.extend(v)

    tables = ','.join(
        f'{conn.index_schema}.{v} AS {k}'
        for k, v in tables.items())

    where = ('WHERE ' + ' AND '.join(where)) if where else ''
    limit = 'LIMIT %d' % (query.get('limit') or _DEFAULT_LIMIT)

    count_sql = f'SELECT COUNT(DISTINCT FS.*) FROM {tables} {where}'
    count = conn.select_value(count_sql, parms)

    data_sql = f'SELECT DISTINCT FS.* FROM {tables} {where} {limit}'
    gws.log.debug(f'sql={data_sql!r} parms={parms!r}')

    data = conn.select(data_sql, parms)

    def _unpack(fs):
        for k in 'lage', 'buchung', 'nutzung', 'gebaeude':
            fs[k] = indexer.from_json(fs.get(k))
        return gws.compact(fs)

    return count, [_unpack(fs) for fs in data]


def _query_to_dict(query):
    return {
        k: v
        for k, v in vars(query).items()
        if v not in (None, '')
    }
