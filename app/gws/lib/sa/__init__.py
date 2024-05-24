"""Convenience wrapper for sqlalchemy imports."""
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
