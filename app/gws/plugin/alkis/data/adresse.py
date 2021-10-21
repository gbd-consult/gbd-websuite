import collections
import re

import gws
import gws.types as t
from gws.lib.console import ProgressIndicator
from . import resolver
from ..util import indexer
from ..util.connection import AlkisConnection

addr_index = 'idx_adresse'
gebs_index = 'idx_gebaeude'

MIN_GEBAEUDE_AREA = 0.5


def normalize_hausnummer(hn):
    # "12 a" -> "12a"
    return re.sub(r'\s+', '', hn.strip())


def int_hausnummer(hn):
    m = re.match(r'^(\d+)', str(hn or '').strip())
    if not m:
        return None
    return int(m.group(1))


def street_name_key(s):
    s = s.strip().lower()

    s = s.replace(u'ä', 'ae')
    s = s.replace(u'ë', 'ee')
    s = s.replace(u'ö', 'oe')
    s = s.replace(u'ü', 'ue')
    s = s.replace(u'ß', 'ss')

    s = re.sub(r'\W+', ' ', s)

    # s = re.sub(r'(?<=\d)\s+', '', s)
    # s = re.sub(r'\s+(?=\d)', '', s)

    s = re.sub(r'\s?str\.$', '.strasse', s)
    s = re.sub(r'\s?pl\.$', '.platz', s)
    s = re.sub(r'\s?(strasse|allee|damm|gasse|pfad|platz|ring|steig|wall|weg|zeile)$', r'.\1', s)

    s = s.replace(' ', '.')

    return s


_lage_tables = (
    'ax_lagebezeichnungohnehausnummer',
    'ax_lagebezeichnungmithausnummer',
    'ax_lagebezeichnungmitpseudonummer')


def _place_key(r):
    # schluesselgesamt should be equal to land+regierungsbezirk+kreis+gemeinde+lage
    # but sometimes it is not... so lets use our own key
    return r['land'], r['regierungsbezirk'], r['kreis'], r['gemeinde'], r['lage']


def _collect_gebs(conn):
    rs = conn.select_from_ax('ax_gebaeude', [
        'gml_id',
        'gebaeudefunktion',
        'weiteregebaeudefunktion',
        'name',
        'bauweise',
        'anzahlderoberirdischengeschosse',
        'anzahlderunterirdischengeschosse',
        'hochhaus',
        'objekthoehe',
        'dachform',
        'zustand',
        'geschossflaeche',
        'grundflaeche',
        'umbauterraum',
        'lagezurerdoberflaeche',
        'dachart',
        'dachgeschossausbau',
        'description',
        'art',
        'individualname',
        'baujahr',
        'wkb_geometry'
    ])

    gebs = []

    for r in rs:
        r = gws.compact(r)
        r.update(resolver.attributes(conn, 'ax_gebaeude', r))
        gebs.append({
            'gml_id': r.pop('gml_id'),
            'geom': r.pop('wkb_geometry'),
            'attributes': indexer.as_json(r)

        })

    return gebs


def _create_gebs_index(conn: AlkisConnection):
    fsx_temp = '_temp_geb_fsx'
    geb_temp = '_temp_geb_geb'

    dat = conn.data_schema
    idx = conn.index_schema

    gws.log.info('gebaeude: copying')

    conn.create_index_table(fsx_temp, f'''
        id SERIAL PRIMARY KEY,
        gml_id CHARACTER VARYING,
        isvalid BOOLEAN,
        geom geometry(GEOMETRY, {conn.srid})
    ''')

    sql = conn.make_select_from_ax('ax_flurstueck', ['gml_id', 'wkb_geometry'])
    conn.exec(f'INSERT INTO {idx}.{fsx_temp} (gml_id, geom) {sql}')

    gebs = _collect_gebs(conn)

    conn.create_index_table(geb_temp, f'''
        id SERIAL PRIMARY KEY,
        gml_id CHARACTER VARYING,
        attributes CHARACTER VARYING,
        isvalid BOOLEAN,
        geom geometry(GEOMETRY, {conn.srid})
    ''')
    conn.index_insert(geb_temp, gebs)

    conn.create_index_index(fsx_temp, 'geom', 'gist')
    conn.create_index_index(geb_temp, 'geom', 'gist')

    gws.log.info('gebaeude: validating')

    conn.validate_index_geoms(geb_temp)
    conn.validate_index_geoms(fsx_temp)

    cnt = conn.count(f'{idx}.{geb_temp}')
    step = 1000

    conn.create_index_table(gebs_index, f'''
        id SERIAL PRIMARY KEY,
        gml_id CHARACTER VARYING,
        fs_id CHARACTER VARYING,
        attributes CHARACTER VARYING,
        area FLOAT,
        fs_geom geometry(GEOMETRY, {conn.srid}),
        gb_geom geometry(GEOMETRY, {conn.srid})
    ''')

    with ProgressIndicator('gebaeude: search', cnt) as pi:
        for n in range(0, cnt, step):
            n1 = n + step
            conn.exec(f'''
                INSERT INTO {idx}.{gebs_index} 
                        (gml_id, fs_id, attributes, fs_geom, gb_geom)
                    SELECT
                        gb.gml_id,
                        fs.gml_id,
                        gb.attributes,
                        fs.geom,
                        gb.geom
                    FROM
                        {idx}.{geb_temp} AS gb,
                        {idx}.{fsx_temp} AS fs
                    WHERE
                        {n} < gb.id AND gb.id <= {n1}
                        AND ST_Intersects(gb.geom, fs.geom)
            ''')
            pi.update(step)

    cnt = conn.count(f'{idx}.{gebs_index}')
    step = 1000

    with ProgressIndicator('gebaeude: areas', cnt) as pi:
        for n in range(0, cnt, step):
            n1 = n + step
            conn.exec(f'''
                UPDATE {idx}.{gebs_index}
                    SET area = ST_Area(ST_Intersection(fs_geom, gb_geom))
                    WHERE
                        {n} < id AND id <= {n1}
            ''')
            pi.update(step)

    gws.log.info('gebaeude: cleaning up')

    conn.exec(f'DELETE FROM {idx}.{gebs_index} WHERE area < %s', [MIN_GEBAEUDE_AREA])
    conn.exec(f'DROP TABLE {idx}.{fsx_temp} CASCADE')
    conn.exec(f'DROP TABLE {idx}.{geb_temp} CASCADE')

    conn.mark_index_table(gebs_index)


def _create_addr_index(conn: AlkisConnection):
    dat = conn.data_schema
    idx = conn.index_schema

    gws.log.info('adresse: reading')

    rs = conn.select_from_ax('ax_lagebezeichnungkatalogeintrag')

    lage_catalog = {}

    for r in rs:
        lage_catalog[_place_key(r)] = [r['gml_id'], r['schluesselgesamt'], r['bezeichnung']]

    lage = {}

    gws.log.info('adresse: collecting')

    for tab in _lage_tables:
        rs = conn.select_from_ax(tab)

        for r in rs:
            if r['unverschluesselt']:
                r['lage_id'] = ''
                r['lage_schluesselgesamt'] = ''
                r['strasse'] = r['unverschluesselt']
            else:
                lg = lage_catalog.get(_place_key(r))
                if lg:
                    r['lage_id'] = lg[0]
                    r['lage_schluesselgesamt'] = lg[1]
                    r['strasse'] = lg[2]

            if 'strasse' not in r or r['strasse'] == 'ohne Lage':
                continue

            for hnr in 'hausnummer', 'pseudonummer', 'laufendenummer':
                if r.get(hnr):
                    r['hausnummer'] = normalize_hausnummer(r[hnr])
                    r['hausnummer_type'] = hnr
                    break

            if not r.get('hausnummer'):
                r['hausnummer'] = ''
                r['hausnummer_type'] = ''

            lage[r['gml_id']] = r

    rs = conn.select_from_ax('ax_flurstueck', [
        'gml_id',
        'weistauf',
        'zeigtauf',
        'land',
        'gemarkungsnummer',
        'gemeinde',
        'regierungsbezirk',
        'kreis',
        'ST_X(ST_Centroid(wkb_geometry)) AS x',
        'ST_Y(ST_Centroid(wkb_geometry)) AS y'
    ])

    exclude_gemarkung = set(conn.exclude_gemarkung)

    total = conn.count(f'{dat}.ax_flurstueck')

    with ProgressIndicator('adresse: index', total) as pi:
        for fs in rs:
            if fs.get('gemarkungsnummer') in exclude_gemarkung:
                continue

            fs_id = fs.pop('gml_id')

            for lage_id in (fs['zeigtauf'] or []) + (fs['weistauf'] or []):

                if lage_id not in lage:
                    continue

                la = lage[lage_id]

                if 'fs_ids' not in la:
                    la['fs_ids'] = []
                la['fs_ids'].append(fs_id)

                if not la.get('gemarkungsnummer'):
                    la.update(fs)

                la['x'] = fs['x']
                la['y'] = fs['y']
                la['xysrc'] = 'fs'

            pi.update(1)

    gws.log.info('adresse: coordinates')

    rs = conn.select_from_ax(
        'ap_pto',
        [
            'dientzurdarstellungvon',
            'ST_X(ST_Centroid(wkb_geometry)) AS x',
            'ST_Y(ST_Centroid(wkb_geometry)) AS y'
        ],
        conditions={
            'art': "?? ='HNR'",
            'endet': "?? IS NULL"
        }
    )

    for r in rs:
        for lage_id in (r['dientzurdarstellungvon'] or []):
            if lage_id in lage:
                lage[lage_id]['x'] = r['x']
                lage[lage_id]['y'] = r['y']
                lage[lage_id]['xysrc'] = 'pto'

    gws.log.info('adresse: normalize gemarkung')

    gg = collections.defaultdict(set)

    for la in lage.values():
        la.update(resolver.places(conn, la))
        if 'gemarkung' in la:
            gg[la['gemarkung']].add(la['gemeinde'])

    gu = {}

    for gemarkung, gemeinde_list in gg.items():
        if len(gemeinde_list) < 2:
            continue
        for gemeinde in gemeinde_list:
            gu[gemarkung, gemeinde] = '%s (%s)' % (gemarkung, gemeinde.replace('Stadt ', ''))

    for la in lage.values():
        if 'gemarkung' in la:
            k = la['gemarkung'], la['gemeinde']
            if k in gu:
                la['gemarkung_v'] = gu[k]
            else:
                la['gemarkung_v'] = la['gemarkung']

    gws.log.info('adresse: normalize strasse')

    for la in lage.values():
        if 'strasse' in la:
            la['strasse'] = re.sub(r'\s+', ' ', la['strasse'].strip())
            la['strasse_k'] = street_name_key(la['strasse'])
            la['hausnummer_n'] = int_hausnummer(la.get('hausnummer'))

    la_buf = []

    for la in lage.values():
        if 'fs_ids' in la:
            for fs_id in la['fs_ids']:
                d = dict(la)
                d['fs_id'] = fs_id
                la_buf.append(d)

    gws.log.info(f'adresse: writing ({len(la_buf)})')

    conn.create_index_table(addr_index, f'''
        gml_id CHARACTER(16) NOT NULL,
        fs_id  CHARACTER(16),

        land CHARACTER VARYING,
        land_id CHARACTER VARYING,
        regierungsbezirk CHARACTER VARYING,
        regierungsbezirk_id CHARACTER VARYING,
        kreis  CHARACTER VARYING,
        kreis_id CHARACTER VARYING,
        gemeinde CHARACTER VARYING,
        gemeinde_id CHARACTER VARYING,
        gemarkung  CHARACTER VARYING,
        gemarkung_v  CHARACTER VARYING,
        gemarkung_id CHARACTER VARYING,

        strasse CHARACTER VARYING,
        strasse_k CHARACTER VARYING,
        hausnummer CHARACTER VARYING,
        hausnummer_n INTEGER,

        lage_id CHARACTER(16),
        lage_schluesselgesamt  CHARACTER VARYING,

        x FLOAT,
        y FLOAT,
        xysrc  CHARACTER VARYING
    ''')

    conn.index_insert(addr_index, la_buf)
    conn.mark_index_table(addr_index)


def create_index(conn: AlkisConnection):
    if not indexer.check_version(conn, gebs_index):
        _create_gebs_index(conn)
    if not indexer.check_version(conn, addr_index):
        _create_addr_index(conn)


def index_ok(conn: AlkisConnection):
    return indexer.check_version(conn, gebs_index) and indexer.check_version(conn, addr_index)


_DEFAULT_LIMIT = 100


def find(conn: AlkisConnection, query):
    where = []
    parms = []

    for k, v in query.items():

        if k in ('land', 'regierungsbezirk', 'kreis', 'gemeinde', 'gemarkung'):
            where.append('AD.' + k + ' = %s')
            parms.append(v)

        elif k in ('landUid', 'regierungsbezirkUid', 'kreisUid', 'gemeindeUid', 'gemarkungUid'):
            where.append('AD.' + (k.replace('Uid', '_id')) + ' = %s')
            parms.append(v)

        elif k == 'strasse':
            where.append('AD.strasse_k = %s')
            parms.append(street_name_key(v))

            hnr = query.get('hausnummer')

            if hnr == '*':
                where.append('AD.hausnummer IS NOT NULL')
            elif hnr:
                where.append('AD.hausnummer = %s')
                parms.append(normalize_hausnummer(hnr))
            elif query.get('hausnummerNotNull'):
                where.append('AD.hausnummer IS NOT NULL')

    where_str = ('WHERE ' + ' AND '.join(where)) if where else ''
    limit = 'LIMIT %d' % (query.get('limit', _DEFAULT_LIMIT))
    tables = f'{conn.index_schema}.{addr_index} AS AD'

    count_sql = f'SELECT COUNT(DISTINCT AD.*) FROM {tables} {where_str}'
    count = conn.select_value(count_sql, parms)

    data_sql = f'SELECT DISTINCT AD.* FROM {tables} {where_str} {limit}'
    gws.log.debug(f'sql={data_sql!r} parms={parms!r}')

    data = conn.select(data_sql, parms)

    return count, list(data)
