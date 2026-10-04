"""Convenience wrapper for SQLAlchemy imports.

Re-exports everything from ``sqlalchemy``, together with ``sqlalchemy.exc`` as ``exc``,
``sqlalchemy.orm`` as ``orm`` and ``geoalchemy2`` as ``geo``. ``Error`` is an alias
for ``sqlalchemy.exc.SQLAlchemyError``.

Example::

    import gws.lib.sa as sa

    tab = sa.Table('places', sa.MetaData(), sa.Column('id', sa.Integer, primary_key=True))
    geom_col = sa.Column('geom', sa.geo.Geometry('POINT', srid=3857))
    try:
        ...
    except sa.Error:
        ...
"""
import sqlalchemy.exc
# noinspection PyUnresolvedReferences
from sqlalchemy import *
# noinspection PyUnresolvedReferences
import sqlalchemy.exc as exc
# noinspection PyUnresolvedReferences
import sqlalchemy.orm as orm
# noinspection PyUnresolvedReferences
import geoalchemy2 as geo

Error = sqlalchemy.exc.SQLAlchemyError
