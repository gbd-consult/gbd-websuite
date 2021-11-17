"""Wrapper for sqlite3.


The wrapper provides some reasonable defaults for sqlite3 connections and ensures that modify operations do not fail because of locking.

"""

import os
import sqlite3
import time

import gws
import gws.types as t

_LOCK_RETRY_ATTEMPTS = 10
_LOCK_WAIT_TIME = 0.05


class _ConnectionWrapper:
    _conn: sqlite3.Connection

    def __init__(self, database):
        self._database = database

    def __enter__(self):
        self._conn = sqlite3.connect(self._database)
        self._conn.row_factory = sqlite3.Row
        if os.path.isfile(self._database):
            os.chown(self._database, gws.UID, gws.GID)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.commit()
        self._conn.close()

    def execute(self, sql, parameters=None):
        for retry in range(0, _LOCK_RETRY_ATTEMPTS):
            try:
                return self._conn.execute(sql, parameters or [])
            except sqlite3.OperationalError as e:
                if 'locked' in e.args[0]:
                    gws.log.warn(f'SQLITE: db={self._database} retry={retry} {e.args[0]} ')
                    time.sleep(_LOCK_WAIT_TIME)
                else:
                    raise

        return self._conn.execute(sql, parameters)


def connect(database) -> _ConnectionWrapper:
    return _ConnectionWrapper(database)
