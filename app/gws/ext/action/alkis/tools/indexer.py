import datetime
import json
import gws

from ..data import version


def default(o):
    if isinstance(o, datetime.date) or isinstance(o, datetime.datetime):
        return o.isoformat()


def as_json(s):
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
    if ver == version.INDEX:
        gws.log.info('index %r version %s, ok' % (table, ver))
        return True
    gws.log.info('index %r version %s, needs update' % (table, ver))
    return False
