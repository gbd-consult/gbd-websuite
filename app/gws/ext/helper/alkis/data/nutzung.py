"""Interface for Objektbereich:Tatsächliche Nutzung"""

from . import resolver, nutzung_festlegung as nf
from ..util import indexer
from ..util.connection import AlkisConnection

parts_index = f'alkis_nutzung_parts_{resolver.VERSION}'
all_index = f'alkis_nutzung_all_{resolver.VERSION}'


def create_index(conn: AlkisConnection):
    if not indexer.check_version(conn, all_index):
        nf.create_all_index(conn, 'nutzung', all_index, parts_index, resolver.nutzung_tables)
    if not indexer.check_version(conn, parts_index):
        nf.create_parts_index(conn, 'nutzung', all_index, parts_index, resolver.nutzung_tables)


def index_ok(conn: AlkisConnection):
    return indexer.check_version(conn, all_index) and indexer.check_version(conn, parts_index)


def get_all(conn: AlkisConnection):
    idx = conn.index_schema
    return conn.select(f'SELECT * FROM {idx}.{all_index}')
