"""Tests for the sqlitex module."""

import multiprocessing
import os

import gws.lib.sqlitex as sqlitex
import gws.test.util as u


def test_basic_insert_and_select(tmp_path):
    """Test basic insert and select operations."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT,
            bbb TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert a record
    db.insert('t1', {'uid': 111, 'aaa': '111', 'bbb': '222'})

    # Select and verify
    results = db.select('SELECT * FROM t1 WHERE uid = :uid', uid=111)
    assert len(results) == 1
    assert results[0]['uid'] == 111
    assert results[0]['aaa'] == '111'
    assert results[0]['bbb'] == '222'


def test_update_operation(tmp_path):
    """Test update operation."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT,
            bbb TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert a record
    db.insert('t1', {'uid': 111, 'aaa': '111', 'bbb': '222'})

    # Update the record
    db.update('t1', {'aaa': '333', 'bbb': '444'}, uid=111)

    # Verify update
    results = db.select('SELECT * FROM t1 WHERE uid = :uid', uid=111)
    assert len(results) == 1
    assert results[0]['aaa'] == '333'
    assert results[0]['bbb'] == '444'


def test_delete_operation(tmp_path):
    """Test delete operation."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert records
    db.insert('t1', {'uid': 111, 'aaa': '111'})
    db.insert('t1', {'uid': 222, 'aaa': '222'})

    # Delete one record
    db.delete('t1', uid=111)

    # Verify deletion
    results = db.select('SELECT * FROM t1')
    assert len(results) == 1
    assert results[0]['uid'] == 222
    assert results[0]['aaa'] == '222'


def test_execute_statement(tmp_path):
    """Test execute method for DML statements."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT,
            bbb REAL
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Use execute for insert
    db.execute('INSERT INTO t1 (uid, aaa, bbb) VALUES (:uid, :aaa, :bbb)', uid=111, aaa='111', bbb=1.5)

    # Verify
    results = db.select('SELECT * FROM t1 WHERE uid = :uid', uid=111)
    assert len(results) == 1
    assert results[0]['aaa'] == '111'
    assert results[0]['bbb'] == 1.5


def test_auto_init_on_missing_table(tmp_path):
    """Test that init_ddl runs automatically when table doesn't exist."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # First query should trigger init_ddl
    db.insert('t1', {'uid': 111, 'aaa': '111'})

    # Verify it worked
    results = db.select('SELECT * FROM t1')
    assert len(results) == 1
    assert results[0]['aaa'] == '111'


def test_multiple_inserts(tmp_path):
    """Test multiple insert operations."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa INTEGER
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert multiple records
    for i in range(1, 6):
        db.insert('t1', {'uid': i, 'aaa': i * 10})

    # Verify all records
    results = db.select('SELECT * FROM t1 ORDER BY uid')
    assert len(results) == 5
    for i, record in enumerate(results, start=1):
        assert record['uid'] == i
        assert record['aaa'] == i * 10


def test_select_with_parameters(tmp_path):
    """Test select with various parameter bindings."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT,
            bbb REAL
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert test data
    db.insert('t1', {'uid': 111, 'aaa': '111', 'bbb': 111})
    db.insert('t1', {'uid': 222, 'aaa': '111', 'bbb': 222})
    db.insert('t1', {'uid': 333, 'aaa': '222', 'bbb': 333})

    # Test filtering by a text column
    results = db.select('SELECT * FROM t1 WHERE aaa = :aaa', aaa='111')
    assert len(results) == 2

    # Test filtering by a numeric range
    results = db.select('SELECT * FROM t1 WHERE bbb > :bbb', bbb=111)
    assert len(results) == 2


def test_empty_select(tmp_path):
    """Test select that returns no results."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Select from empty table
    results = db.select('SELECT * FROM t1')
    assert len(results) == 0
    assert isinstance(results, list)


def test_update_nonexistent_record(tmp_path):
    """Test updating a record that doesn't exist."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Try to update non-existent record (should not raise error)
    db.update('t1', {'aaa': '111'}, uid=111)

    # Verify nothing was changed
    results = db.select('SELECT * FROM t1')
    assert len(results) == 0


def test_delete_nonexistent_record(tmp_path):
    """Test deleting a record that doesn't exist."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Try to delete non-existent record (should not raise error)
    db.delete('t1', uid=111)

    # Verify table is still empty
    results = db.select('SELECT * FROM t1')
    assert len(results) == 0


def test_without_init_ddl(tmp_path):
    """Test creating database without init_ddl."""
    db_path = tmp_path / 'test.db'

    # Create database without init_ddl
    db = sqlitex.Object(str(db_path))

    # Manually create table
    db.execute("""
        CREATE TABLE IF NOT EXISTS t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """)

    # Insert and verify
    db.insert('t1', {'uid': 111, 'aaa': '111'})
    results = db.select('SELECT * FROM t1')
    assert len(results) == 1


def test_complex_query(tmp_path):
    """Test more complex SQL queries."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT,
            bbb REAL,
            ccc TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert test data
    db.insert('t1', {'uid': 111, 'aaa': '111', 'bbb': 100.50, 'ccc': '111'})
    db.insert('t1', {'uid': 222, 'aaa': '222', 'bbb': 250.75, 'ccc': '222'})
    db.insert('t1', {'uid': 333, 'aaa': '111', 'bbb': 75.25, 'ccc': '111'})

    # Complex query with aggregation
    results = db.select(
        """
        SELECT aaa, SUM(bbb) as ddd, COUNT(*) as eee
        FROM t1
        WHERE ccc = :ccc
        GROUP BY aaa
        ORDER BY ddd DESC
    """,
        ccc='111',
    )

    assert len(results) == 1
    assert results[0]['aaa'] == '111'
    assert results[0]['ddd'] == 175.75
    assert results[0]['eee'] == 2


def test_special_characters_in_data(tmp_path):
    """Test handling of special characters in data."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert data with special characters
    special_text = 'Hello \'world\' with "quotes" and \n newlines \t tabs'
    db.insert('t1', {'uid': 111, 'aaa': special_text})

    # Verify
    results = db.select('SELECT * FROM t1 WHERE uid = :uid', uid=111)
    assert len(results) == 1
    assert results[0]['aaa'] == special_text


def test_null_values(tmp_path):
    """Test handling of NULL values."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT,
            bbb TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert record with NULL
    db.execute('INSERT INTO t1 (uid, aaa, bbb) VALUES (:uid, :aaa, :bbb)', uid=111, aaa='111', bbb=None)

    # Verify
    results = db.select('SELECT * FROM t1 WHERE uid = :uid', uid=111)
    assert len(results) == 1
    assert results[0]['aaa'] == '111'
    assert results[0]['bbb'] is None


def test_reuse_database(tmp_path):
    """Test that database persists and can be reopened."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """

    # First connection
    db1 = sqlitex.Object(str(db_path), init_ddl)
    db1.insert('t1', {'uid': 111, 'aaa': '111'})

    # Second connection to same database
    db2 = sqlitex.Object(str(db_path), init_ddl)
    results = db2.select('SELECT * FROM t1')

    assert len(results) == 1
    assert results[0]['aaa'] == '111'


def test_multi_statement_ddl(tmp_path):
    """Test an init script with several statements."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        );
        CREATE INDEX IF NOT EXISTS t1_aaa ON t1(aaa);
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    db.insert('t1', {'uid': 111, 'aaa': '111'})

    results = db.select("SELECT name FROM sqlite_master WHERE type = 'index' AND name = :name", name='t1_aaa')
    assert len(results) == 1


def test_reinit_after_delete(tmp_path):
    """Test that the database is recreated if the file is removed."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)
    db.insert('t1', {'uid': 111, 'aaa': '111'})

    os.unlink(str(db_path))

    results = db.select('SELECT * FROM t1')
    assert len(results) == 0

    db.insert('t1', {'uid': 222, 'aaa': '222'})
    results = db.select('SELECT * FROM t1')
    assert len(results) == 1
    assert results[0]['aaa'] == '222'


def test_error_is_not_retried(tmp_path):
    """Test that an invalid statement raises immediately."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS t1 (
            uid INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)
    db.insert('t1', {'uid': 111, 'aaa': '111'})

    with u.raises(sqlitex.Error):
        db.select('SELECT zzz FROM t1')


def test_busy_timeout_is_set(tmp_path):
    """Test that the module busy timeout reaches the connection."""
    db_path = tmp_path / 'test.db'

    db = sqlitex.Object(str(db_path))
    assert db.select('PRAGMA busy_timeout')[0]['timeout'] == int(sqlitex.BUSY_TIMEOUT * 1000)


def test_custom_uid_column(tmp_path):
    """Test update and delete with a non-default primary key name."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS t1 (
            zzz INTEGER PRIMARY KEY,
            aaa TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl, uid_column='zzz')

    db.insert('t1', {'zzz': 111, 'aaa': '111'})
    db.insert('t1', {'zzz': 222, 'aaa': '222'})

    db.update('t1', {'aaa': '333'}, uid=111)
    results = db.select('SELECT * FROM t1 WHERE zzz = :zzz', zzz=111)
    assert results[0]['aaa'] == '333'

    db.delete('t1', uid=222)
    results = db.select('SELECT * FROM t1')
    assert len(results) == 1
    assert results[0]['zzz'] == 111


def _mp_worker(n, db_path, num_loops):
    """Worker function for multiprocessing concurrency test."""

    db = sqlitex.Object(str(db_path))

    for i in range(num_loops):
        db.execute(
            """
            UPDATE t1 SET
                aaa = aaa + 1,
                bbb = :bbb
            WHERE uid = 111
        """,
            bbb=n,
        )


def test_concurrency(tmp_path):
    """Test concurrent writes using multiprocessing (true parallelism)."""
    num_processes = 50
    num_loops = 10
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS t1 (
            uid INTEGER PRIMARY KEY,
            aaa INTEGER,
            bbb INTEGER
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)
    db.insert('t1', {'uid': 111, 'aaa': 0, 'bbb': -1})

    ps = []

    for n in range(num_processes):
        p = multiprocessing.Process(target=_mp_worker, args=[n, db_path, num_loops])
        ps.append(p)
        p.start()

    for p in ps:
        p.join()

    assert [p.exitcode for p in ps] == [0] * num_processes

    db = sqlitex.Object(str(db_path))

    results = db.select('SELECT aaa FROM t1 WHERE uid = 111')
    expected_value = num_processes * num_loops
    assert results[0]['aaa'] == expected_value
