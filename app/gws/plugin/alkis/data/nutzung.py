"""Interface for Objektbereich:Tatsächliche Nutzung"""

import gws
import gws.types as t
from gws.lib.console import ProgressIndicator

from . import resolver
from ..util import indexer
from ..util.connection import AlkisConnection

parts_index = 'idx_nutzung_parts'
all_index = 'idx_nutzung_all'

MIN_AREA = 0.01


def _collect_from_nutzung_tables(conn: AlkisConnection):
    tables = conn.table_names(conn.data_schema)
    data = []

    for type_id, table, type in resolver.nutzung_tables:
        if table not in tables:
            continue

        rs = conn.select_from_ax(table)

        for r in rs:
            if 'gml_id' not in r or 'wkb_geometry' not in r:
                continue
            a = resolver.attributes(conn, table, r)
            k = resolver.nutzung_key(type_id, a) or {'key': type, 'key_id': type_id, 'key_label': 'Typ'}
            a.update(k)
            data.append({
                'gml_id': r['gml_id'],
                'type': type,
                'type_id': type_id,
                'geom': r['wkb_geometry'],
                'attributes': indexer.to_json(a)
            })
    return data


def _create_all_index(conn: AlkisConnection):
    data = _collect_from_nutzung_tables(conn)

    conn.create_index_table(all_index, f'''
        id SERIAL PRIMARY KEY,
        gml_id CHARACTER VARYING,
        type CHARACTER VARYING,
        type_id INTEGER,
        attributes CHARACTER VARYING,
        isvalid BOOLEAN,
        geom geometry(GEOMETRY, {conn.srid})
    ''')

    conn.index_insert(all_index, data)

    conn.create_index_index(all_index, 'geom', 'gist')
    conn.create_index_index(all_index, 'gml_id', 'btree')

    gws.log.info('nutzung: validating geometries')
    conn.validate_index_geoms(all_index)

    conn.mark_index_table(all_index)


def _delete_empty_areas(conn: AlkisConnection, col):
    idx = conn.index_schema

    cnt_all = conn.select_value(f'SELECT COUNT(*) FROM {idx}.{parts_index}')
    cnt_nul = conn.select_value(f'SELECT COUNT(*) FROM {idx}.{parts_index} WHERE {col} < {MIN_AREA}')

    gws.log.info(f'nutzung: {cnt_all} areas, {cnt_nul} empty')

    if cnt_nul:
        conn.exec(f'DELETE FROM {idx}.{parts_index} WHERE {col} < {MIN_AREA}')


def _create_parts_index(conn: AlkisConnection):
    idx = conn.index_schema

    fs_temp = '_idx_nutzung_fs_temp'

    conn.create_index_table(fs_temp, f'''
            id SERIAL PRIMARY KEY,
            gml_id CHARACTER VARYING,
            area DOUBLE PRECISION,
            a_area DOUBLE PRECISION,
            area_factor DOUBLE PRECISION,
            isvalid BOOLEAN,
            geom geometry(GEOMETRY, {conn.srid})
    ''')

    sel = conn.make_select_from_ax('ax_flurstueck', [
        'gml_id',
        'ST_Area(wkb_geometry)',
        'amtlicheflaeche',
        'wkb_geometry'
    ])

    conn.exec(f'INSERT INTO {idx}.{fs_temp}(gml_id,area,a_area,geom) {sel}')

    conn.create_index_index(fs_temp, 'geom', 'gist')
    conn.create_index_index(fs_temp, 'gml_id', 'btree')

    conn.validate_index_geoms(fs_temp)

    conn.create_index_table(parts_index, f'''
            id SERIAL PRIMARY KEY,
            fs_id CHARACTER VARYING,
            nu_id CHARACTER VARYING,
            type CHARACTER VARYING,
            type_id INTEGER,
            attributes CHARACTER VARYING,
            area float,
            a_area float,
            fs_geom geometry(GEOMETRY, {conn.srid}),
            nu_geom geometry(GEOMETRY, {conn.srid}),
            part_geom geometry(GEOMETRY, {conn.srid})
    ''')

    gws.log.info('nutzung: all=%d' % conn.count(f'{idx}.{all_index}'))
    max_id = conn.select_value(f'SELECT MAX(id) + 1 FROM {idx}.{all_index}')

    step = 1000

    with ProgressIndicator('nutzung: search', max_id) as pi:
        for n in range(0, max_id, step):
            n1 = n + step
            conn.exec(f'''
                INSERT INTO {idx}.{parts_index} 
                        (fs_id, nu_id, type, type_id, attributes, fs_geom, nu_geom)
                    SELECT
                        fs.gml_id,
                        nu.gml_id,
                        nu.type,
                        nu.type_id,
                        nu.attributes,
                        fs.geom,
                        nu.geom
                    FROM
                        {idx}.{all_index} AS nu,
                        {idx}.{fs_temp} AS fs
                    WHERE
                        {n} < nu.id AND nu.id <= {n1}
                        AND ST_Intersects(nu.geom, fs.geom)
            ''')
            pi.update(step)

    gws.log.info('nutzung: parts=%d' % conn.count(f'{idx}.{parts_index}'))
    max_id = conn.select_value(f'SELECT MAX(id) + 1 FROM {idx}.{parts_index}')
    step = 1000

    with ProgressIndicator('nutzung: intersections', max_id) as pi:
        for n in range(0, max_id, step):
            n1 = n + step
            conn.exec(f'''
                UPDATE {idx}.{parts_index} AS nu
                    SET part_geom = ST_Intersection(fs_geom, nu_geom)
                    WHERE {n} < nu.id AND nu.id <= {n1}
            ''')
            pi.update(step)

    with ProgressIndicator('nutzung: areas', max_id) as pi:
        for n in range(0, max_id, step):
            n1 = n + step
            conn.exec(f'''
                UPDATE {idx}.{parts_index} AS nu
                    SET area = ST_Area(part_geom)
                    WHERE {n} < nu.id AND nu.id <= {n1}
            ''')
            pi.update(step)

    _delete_empty_areas(conn, 'area')

    # compute "amtliche" nutzung areas
    # see norbit/alkisimport/alkis-nutzung-und-klassifizierung.sql:445
    # nu[a_area] = nu[area] * (fs[a_area] / fs[area])

    gws.log.info('nutzung: correcting areas')

    gws.log.info('nutzung: parts=%d' % conn.count(f'{idx}.{parts_index}'))
    max_id = conn.select_value(f'SELECT MAX(id) + 1 FROM {idx}.{parts_index}')
    step = 1000

    conn.exec(f'UPDATE {idx}.{fs_temp} SET area_factor = a_area / area')

    with ProgressIndicator('nutzung: correcting', max_id) as pi:
        for n in range(0, max_id, step):
            n1 = n + step
            conn.exec(f'''
                UPDATE {idx}.{parts_index} AS nu
                    SET a_area = nu.area * fs.area_factor
                    FROM {idx}.{fs_temp} AS fs
                    WHERE {n} < nu.id AND nu.id <= {n1}
                        AND nu.fs_id = fs.gml_id
            ''')
            pi.update(step)

    _delete_empty_areas(conn, 'a_area')

    gws.log.info('nutzung: parts=%d' % conn.count(f'{idx}.{parts_index}'))

    conn.exec(f'DROP TABLE IF EXISTS {idx}.{fs_temp} CASCADE')
    conn.mark_index_table(parts_index)


def create_index(conn: AlkisConnection):
    if not indexer.check_version(conn, all_index):
        _create_all_index(conn)
    if not indexer.check_version(conn, parts_index):
        _create_parts_index(conn)


def index_ok(conn: AlkisConnection):
    return indexer.check_version(conn, all_index) and indexer.check_version(conn, parts_index)


def get_all(conn: AlkisConnection):
    idx = conn.index_schema
    return conn.select(f'SELECT * FROM {idx}.{all_index}')
