import gws
import gws.types as t

from . import resolver, adresse, nutzung, grundbuch, flurstueck
from ..util.connection import AlkisConnection

mods = [resolver, adresse, nutzung, grundbuch, flurstueck]


def create(conn: AlkisConnection, read_user):
    resolver.create_index(conn)
    adresse.create_index(conn)
    nutzung.create_index(conn)
    grundbuch.create_index(conn)
    flurstueck.create_index(conn)

    idx = conn.index_schema

    for tab in conn.table_names(idx):
        gws.log.info(f'optimizing {tab!r}')
        conn.exec(f'VACUUM {idx}.{tab}')
    conn.exec(f'GRANT SELECT  ON ALL TABLES    IN SCHEMA "{idx}" TO "{read_user}"')
    conn.exec(f'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA "{idx}" TO "{read_user}"')


def ok(conn: AlkisConnection):
    return all([
        resolver.index_ok(conn),
        adresse.index_ok(conn),
        nutzung.index_ok(conn),
        grundbuch.index_ok(conn),
        flurstueck.index_ok(conn),
    ])


def drop(conn: AlkisConnection):
    conn.drop_all_indexes()
