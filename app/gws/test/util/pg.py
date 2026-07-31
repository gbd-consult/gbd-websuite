"""Postgres test database."""

from typing import Optional

import gws
import gws.base.shape
import gws.lib.crs
import gws.lib.net
import gws.lib.sa as sa

from . import options

_engine: Optional[sa.Engine] = None


def connect():
    global _engine

    if not _engine:
        _engine = sa.create_engine(url(), poolclass=sa.NullPool)

    return _engine.connect()


def url():
    return gws.lib.net.make_url(
        scheme='postgresql',
        username=options.option('service.postgres.user'),
        password=options.option('service.postgres.password'),
        hostname=options.option('service.postgres.host'),
        port=options.option('service.postgres.port'),
        path=options.option('service.postgres.database'),
    )


def provider(root: gws.Root) -> gws.DatabaseProvider:
    return root.get('GWS_TEST_POSTGRES_PROVIDER')


def create_schema(name: str):
    with connect() as conn:
        conn.execute(sa.text(f'DROP SCHEMA IF EXISTS {name} CASCADE'))
        conn.execute(sa.text(f'CREATE SCHEMA {name}'))
        conn.commit()


def create(table_name: str, col_defs: dict):
    with connect() as conn:
        conn.execute(sa.text(f'DROP TABLE IF EXISTS {table_name} CASCADE'))
        ddl = _comma(f'{k} {v}' for k, v in col_defs.items())
        conn.execute(sa.text(f'CREATE TABLE {table_name} ( {ddl} )'))
        conn.commit()


def clear(table_name: str):
    with connect() as conn:
        conn.execute(sa.text(f'TRUNCATE TABLE {table_name}'))
        conn.commit()


def insert(table_name: str, row_dicts: list[dict]):
    with connect() as conn:
        conn.execute(sa.text(f'TRUNCATE TABLE {table_name}'))
        if row_dicts:
            names = _comma(k for k in row_dicts[0])
            values = _comma(':' + k for k in row_dicts[0])
            ins = sa.text(f'INSERT INTO {table_name} ( {names} ) VALUES( {values} )')
            conn.execute(ins, row_dicts)
        conn.commit()


def rows(sql: str) -> list[tuple]:
    with connect() as conn:
        return [tuple(r) for r in conn.execute(sa.text(sql))]


def content(sql_or_table_name: str) -> list[tuple]:
    if not sql_or_table_name.lower().startswith('select'):
        sql_or_table_name = f'SELECT * FROM {sql_or_table_name}'
    return rows(sql_or_table_name)


def exec(sql: str, **kwargs):
    with connect() as conn:
        for s in sql.split(';'):
            if s.strip():
                conn.execute(sa.text(s.strip()), kwargs)
        conn.commit()


def connections():
    with connect() as conn:
        mark = gws.u.random_string(16)
        sql = f"""
        SELECT
            *,
            '{mark}' AS _mark
        FROM
            pg_stat_activity
        WHERE
            backend_type = 'client backend'
        """

        rs = []
        for r in conn.execute(sa.text(sql)):
            r = r._asdict()
            r.pop('_mark', None)
            q = r.get('query', '')
            if mark in q:
                continue
            rs.append(r)
        return rs


def ewkb(wkt: str, srid=3857):
    shape = gws.base.shape.from_wkt(wkt, default_crs=gws.lib.crs.get(srid))
    return shape.to_ewkb()


_comma = ','.join
