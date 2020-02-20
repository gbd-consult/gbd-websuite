import gws

from . import resolver, adresse, nutzung, grundbuch, flurstueck
from ..util.connection import AlkisConnection

mods = [resolver, adresse, nutzung, grundbuch, flurstueck]


def create(conn: AlkisConnection, read_user):
    for mod in mods:
        mod.create_index(conn)
    for tab in conn.table_names(conn.index_schema):
        gws.log.info(f'optimizing {tab!r}')
        conn.exec(f'VACUUM {conn.index_schema}.{tab}')
    conn.exec(f'GRANT SELECT  ON ALL TABLES    IN SCHEMA {conn.index_schema} TO {read_user}')
    conn.exec(f'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA {conn.index_schema} TO {read_user}')


def ok(conn: AlkisConnection):
    return all(mod.index_ok(conn) for mod in mods)


def drop(conn: AlkisConnection):
    conn.drop_all()
