"""Generic SQL and SQLAlchemy tools."""

import gws
import gws.types as t

import sqlalchemy as sa
import sqlalchemy.orm as orm
import sqlalchemy.exc as exc
import geoalchemy2 as geosa

__all__ = ['sa', 'orm', 'exc', 'geosa']


class Session(gws.IDatabaseSession):
    saSession: orm.Session

    def __init__(self, sess):
        self.saSession = sess

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        self.saSession.close()

    def begin(self):
        return self.saSession.begin()

    def commit(self):
        return self.saSession.commit()

    def rollback(self):
        return self.saSession.rollback()


class SelectStatement(gws.Data):
    saSelect: sa.select
    search: gws.SearchArgs
    keywordWhere: list
    geometryWhere: list


ATTR_TO_SA = {
    gws.AttributeType.bool: sa.Boolean,
    gws.AttributeType.date: sa.Date,
    gws.AttributeType.datetime: sa.DateTime,
    gws.AttributeType.float: sa.Float,
    gws.AttributeType.int: sa.Integer,
    gws.AttributeType.str: sa.String,
    gws.AttributeType.time: sa.Time,
    gws.AttributeType.geometry: geosa.Geometry,
}

# http://initd.org/psycopg/docs/usage.html?highlight=smallint#adaptation-of-python-values-to-sql-types

SA_TO_ATTR = {
    'ARRAY': gws.AttributeType.strlist,
    'BIGINT': gws.AttributeType.int,
    'BIGSERIAL': gws.AttributeType.int,
    'BIT': gws.AttributeType.int,
    'BOOL': gws.AttributeType.bool,
    'BOOLEAN': gws.AttributeType.bool,
    'BYTEA': gws.AttributeType.bytes,
    'CHAR': gws.AttributeType.str,
    'CHARACTER VARYING': gws.AttributeType.str,
    'CHARACTER': gws.AttributeType.str,
    'DATE': gws.AttributeType.date,
    'DECIMAL': gws.AttributeType.float,
    'DOUBLE PRECISION': gws.AttributeType.float,
    'FLOAT4': gws.AttributeType.float,
    'FLOAT8': gws.AttributeType.float,
    'GEOMETRY': gws.AttributeType.geometry,
    'INT': gws.AttributeType.int,
    'INT2': gws.AttributeType.int,
    'INT4': gws.AttributeType.int,
    'INT8': gws.AttributeType.int,
    'INTEGER': gws.AttributeType.int,
    'MONEY': gws.AttributeType.float,
    'NUMERIC': gws.AttributeType.float,
    'REAL': gws.AttributeType.float,
    'SERIAL': gws.AttributeType.int,
    'SERIAL2': gws.AttributeType.int,
    'SERIAL4': gws.AttributeType.int,
    'SERIAL8': gws.AttributeType.int,
    'SMALLINT': gws.AttributeType.int,
    'SMALLSERIAL': gws.AttributeType.int,
    'TEXT': gws.AttributeType.str,
    'TIME': gws.AttributeType.time,
    'TIMESTAMP': gws.AttributeType.datetime,
    'TIMESTAMPTZ': gws.AttributeType.datetime,
    'TIMETZ': gws.AttributeType.time,
    'VARCHAR': gws.AttributeType.str,
}


def escape_like(s, escape='\\'):
    return (
        s
        .replace(escape, escape + escape)
        .replace('%', escape + '%')
        .replace('_', escape + '_'))
