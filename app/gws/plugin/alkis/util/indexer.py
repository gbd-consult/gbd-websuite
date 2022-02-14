import datetime
import json

import gws
import gws.types as t

from . import connection


def default(o):
    if isinstance(o, datetime.date) or isinstance(o, datetime.datetime):
        return o.isoformat()


def to_json(s):
    if s is None:
        return None
    return json.dumps(s, ensure_ascii=False, indent=4, sort_keys=True, default=default)


def from_json(s):
    if s is None:
        return None
    return json.loads(s)


def validate_geoms(conn, table):
    warnings = conn.validate_index_geoms(table)
    for w in warnings:
        gws.log.warn('geometry error in %r: %s' % (table, w))


def check_version(conn, table):
    ver = conn.index_table_version(table)
    if ver == connection.INDEX_VERSION:
        gws.log.debug('index %r version %s, ok' % (table, ver))
        return True
    gws.log.warn('index %r version %s, needs update' % (table, ver))
    return False
