import re

import gws

from . import resolver, adresse, nutzung, festlegung, grundbuch, flurstueck, version
from ..util.connection import AlkisConnection

mods = [resolver, adresse, nutzung, festlegung, grundbuch, flurstueck]

_index_table_re = f'^alkis_.+?_{resolver.VERSION}$'


def create(conn: AlkisConnection, read_user):
    for mod in mods:
        mod.create_index(conn)
    for tab_name in conn.table_names(conn.index_schema):
        if re.match(_index_table_re, tab_name):
            gws.log.info(f'optimizing {tab_name!r}')
            conn.exec(f'VACUUM {conn.index_schema}.{tab_name}')
    conn.exec(f'GRANT SELECT  ON ALL TABLES    IN SCHEMA "{conn.index_schema}" TO "{read_user}"')
    conn.exec(f'GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA "{conn.index_schema}" TO "{read_user}"')


def ok(conn: AlkisConnection):
    return all(mod.index_ok(conn) for mod in mods)


def drop(conn: AlkisConnection):
    for tab_name in conn.table_names(conn.index_schema):
        if re.match(_index_table_re, tab_name):
            conn.exec(f'DROP TABLE IF EXISTS {conn.index_schema}.{tab_name}')
