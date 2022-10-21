import psycopg2
import psycopg2.extras

import gws

from . import features, util
from .glob import CONFIG


def connect_params():
    return {
        'database': CONFIG['service.postgres.database'],
        'user': CONFIG['service.postgres.user'],
        'password': CONFIG['service.postgres.password'],
        'port': CONFIG['service.postgres.port'],
        'host': CONFIG['runner.host_name'],
    }


def connection():
    return psycopg2.connect(**connect_params())


def make_features(name, geom_type, columns, crs, xy, rows, cols, gap):
    colnames = list(columns)
    coldefs = [f'{c} {columns[c]}' for c in colnames]

    fs = features.make(name, geom_type, columns, crs, xy, rows, cols, gap)
    shape = fs[0].shape
    if shape:
        colnames.append('p_geom')
        coldefs.append(f'p_geom GEOMETRY({shape.type},{shape.srid})')

    data = []
    for f in fs:
        rec = [a.value for a in f.attributes]
        if f.shape:
            rec.append(f.shape.ewkb_hex)
        data.append(rec)

    conn = connection()
    cur = conn.cursor()

    cur.execute(f'BEGIN')
    cur.execute(f'DROP TABLE IF EXISTS {name}')
    cur.execute(f'''
        CREATE TABLE {name} (
            id SERIAL PRIMARY KEY,
            {','.join(coldefs)}
        )
    ''')
    cur.execute(f'COMMIT')

    cur.execute(f'BEGIN')
    ins = f'''INSERT INTO {name} ({','.join(colnames)}) VALUES %s'''
    psycopg2.extras.execute_values(cur, ins, data)
    cur.execute(f'COMMIT')

    conn.close()


def drop_table(name):
    conn = connection()
    cur = conn.cursor()

    cur.execute(f'BEGIN')
    cur.execute(f'DROP TABLE IF EXISTS {name}')
    cur.execute(f'COMMIT')
    conn.close()
