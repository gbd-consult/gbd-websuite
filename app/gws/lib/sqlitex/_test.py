"""Tests for the sqlitex module."""

import multiprocessing
import os

import gws.lib.sqlitex as sqlitex
import gws.test.util as u


def test_basic_insert_and_select(tmp_path):
    """Test basic insert and select operations."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT,
            col_2 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert a record
    db.insert('table_1', {'uid': 1, 'col_1': 'value_1', 'col_2': 'value_2'})

    # Select and verify
    results = db.select('SELECT * FROM table_1 WHERE uid = :uid', uid=1)
    assert len(results) == 1
    assert results[0]['uid'] == 1
    assert results[0]['col_1'] == 'value_1'
    assert results[0]['col_2'] == 'value_2'


def test_update_operation(tmp_path):
    """Test update operation."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT,
            col_2 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert a record
    db.insert('table_1', {'uid': 1, 'col_1': 'value_1', 'col_2': 'value_2'})

    # Update the record
    db.update('table_1', {'col_1': 'value_3', 'col_2': 'value_4'}, uid=1)

    # Verify update
    results = db.select('SELECT * FROM table_1 WHERE uid = :uid', uid=1)
    assert len(results) == 1
    assert results[0]['col_1'] == 'value_3'
    assert results[0]['col_2'] == 'value_4'


def test_delete_operation(tmp_path):
    """Test delete operation."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert records
    db.insert('table_1', {'uid': 1, 'col_1': 'value_1'})
    db.insert('table_1', {'uid': 2, 'col_1': 'value_2'})

    # Delete one record
    db.delete('table_1', uid=1)

    # Verify deletion
    results = db.select('SELECT * FROM table_1')
    assert len(results) == 1
    assert results[0]['uid'] == 2
    assert results[0]['col_1'] == 'value_2'


def test_execute_statement(tmp_path):
    """Test execute method for DML statements."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT,
            col_2 REAL
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Use execute for insert
    db.execute('INSERT INTO table_1 (uid, col_1, col_2) VALUES (:uid, :col_1, :col_2)', uid=1, col_1='value_1', col_2=1.5)

    # Verify
    results = db.select('SELECT * FROM table_1 WHERE uid = :uid', uid=1)
    assert len(results) == 1
    assert results[0]['col_1'] == 'value_1'
    assert results[0]['col_2'] == 1.5


def test_auto_init_on_missing_table(tmp_path):
    """Test that init_ddl runs automatically when table doesn't exist."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # First query should trigger init_ddl
    db.insert('table_1', {'uid': 1, 'col_1': 'value_1'})

    # Verify it worked
    results = db.select('SELECT * FROM table_1')
    assert len(results) == 1
    assert results[0]['col_1'] == 'value_1'


def test_multiple_inserts(tmp_path):
    """Test multiple insert operations."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 INTEGER
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert multiple records
    for i in range(1, 6):
        db.insert('table_1', {'uid': i, 'col_1': i * 10})

    # Verify all records
    results = db.select('SELECT * FROM table_1 ORDER BY uid')
    assert len(results) == 5
    for i, record in enumerate(results, start=1):
        assert record['uid'] == i
        assert record['col_1'] == i * 10


def test_select_with_parameters(tmp_path):
    """Test select with various parameter bindings."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT,
            col_2 REAL
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert test data
    db.insert('table_1', {'uid': 1, 'col_1': 'value_1', 'col_2': 10})
    db.insert('table_1', {'uid': 2, 'col_1': 'value_1', 'col_2': 20})
    db.insert('table_1', {'uid': 3, 'col_1': 'value_2', 'col_2': 30})

    # Test filtering by a text column
    results = db.select('SELECT * FROM table_1 WHERE col_1 = :col_1', col_1='value_1')
    assert len(results) == 2

    # Test filtering by a numeric range
    results = db.select('SELECT * FROM table_1 WHERE col_2 > :col_2', col_2=10)
    assert len(results) == 2


def test_empty_select(tmp_path):
    """Test select that returns no results."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Select from empty table
    results = db.select('SELECT * FROM table_1')
    assert len(results) == 0
    assert isinstance(results, list)


def test_update_nonexistent_record(tmp_path):
    """Test updating a record that doesn't exist."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Try to update non-existent record (should not raise error)
    db.update('table_1', {'col_1': 'value_1'}, uid=1)

    # Verify nothing was changed
    results = db.select('SELECT * FROM table_1')
    assert len(results) == 0


def test_delete_nonexistent_record(tmp_path):
    """Test deleting a record that doesn't exist."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Try to delete non-existent record (should not raise error)
    db.delete('table_1', uid=1)

    # Verify table is still empty
    results = db.select('SELECT * FROM table_1')
    assert len(results) == 0


def test_without_init_ddl(tmp_path):
    """Test creating database without init_ddl."""
    db_path = tmp_path / 'test.db'

    # Create database without init_ddl
    db = sqlitex.Object(str(db_path))

    # Manually create table
    db.execute("""
        CREATE TABLE IF NOT EXISTS table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """)

    # Insert and verify
    db.insert('table_1', {'uid': 1, 'col_1': 'value_1'})
    results = db.select('SELECT * FROM table_1')
    assert len(results) == 1


def test_complex_query(tmp_path):
    """Test more complex SQL queries."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT,
            col_2 REAL,
            col_3 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert test data
    db.insert('table_1', {'uid': 1, 'col_1': 'value_1', 'col_2': 100.50, 'col_3': 'value_3'})
    db.insert('table_1', {'uid': 2, 'col_1': 'value_2', 'col_2': 250.75, 'col_3': 'value_4'})
    db.insert('table_1', {'uid': 3, 'col_1': 'value_1', 'col_2': 75.25, 'col_3': 'value_3'})

    # Complex query with aggregation
    results = db.select(
        """
        SELECT col_1, SUM(col_2) as col_4, COUNT(*) as col_5
        FROM table_1
        WHERE col_3 = :col_3
        GROUP BY col_1
        ORDER BY col_4 DESC
    """,
        col_3='value_3',
    )

    assert len(results) == 1
    assert results[0]['col_1'] == 'value_1'
    assert results[0]['col_4'] == 175.75
    assert results[0]['col_5'] == 2


def test_special_characters_in_data(tmp_path):
    """Test handling of special characters in data."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert data with special characters
    special_text = 'Hello \'world\' with "quotes" and \n newlines \t tabs'
    db.insert('table_1', {'uid': 1, 'col_1': special_text})

    # Verify
    results = db.select('SELECT * FROM table_1 WHERE uid = :uid', uid=1)
    assert len(results) == 1
    assert results[0]['col_1'] == special_text


def test_null_values(tmp_path):
    """Test handling of NULL values."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT,
            col_2 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert record with NULL
    db.execute('INSERT INTO table_1 (uid, col_1, col_2) VALUES (:uid, :col_1, :col_2)', uid=1, col_1='value_1', col_2=None)

    # Verify
    results = db.select('SELECT * FROM table_1 WHERE uid = :uid', uid=1)
    assert len(results) == 1
    assert results[0]['col_1'] == 'value_1'
    assert results[0]['col_2'] is None


def test_reuse_database(tmp_path):
    """Test that database persists and can be reopened."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """

    # First connection
    db1 = sqlitex.Object(str(db_path), init_ddl)
    db1.insert('table_1', {'uid': 1, 'col_1': 'value_1'})

    # Second connection to same database
    db2 = sqlitex.Object(str(db_path), init_ddl)
    results = db2.select('SELECT * FROM table_1')

    assert len(results) == 1
    assert results[0]['col_1'] == 'value_1'


def test_multi_statement_ddl(tmp_path):
    """Test an init script with several statements."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        );
        CREATE INDEX IF NOT EXISTS index_1 ON table_1(col_1);
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    db.insert('table_1', {'uid': 1, 'col_1': 'value_1'})

    results = db.select("SELECT name FROM sqlite_master WHERE type = 'index' AND name = :name", name='index_1')
    assert len(results) == 1


def test_reinit_after_delete(tmp_path):
    """Test that the database is recreated if the file is removed."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)
    db.insert('table_1', {'uid': 1, 'col_1': 'value_1'})

    os.unlink(str(db_path))

    results = db.select('SELECT * FROM table_1')
    assert len(results) == 0

    db.insert('table_1', {'uid': 2, 'col_1': 'value_2'})
    results = db.select('SELECT * FROM table_1')
    assert len(results) == 1
    assert results[0]['col_1'] == 'value_2'


def test_error_is_not_retried(tmp_path):
    """Test that an invalid statement raises immediately."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)
    db.insert('table_1', {'uid': 1, 'col_1': 'value_1'})

    with u.raises(sqlitex.Error):
        db.select('SELECT col_9 FROM table_1')


def test_busy_timeout_is_set(tmp_path):
    """Test that the module busy timeout reaches the connection."""
    db_path = tmp_path / 'test.db'

    db = sqlitex.Object(str(db_path))
    assert db.select('PRAGMA busy_timeout')[0]['timeout'] == int(sqlitex.BUSY_TIMEOUT * 1000)


def test_custom_uid_column(tmp_path):
    """Test update and delete with a non-default primary key name."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS table_1 (
            key_1 INTEGER PRIMARY KEY,
            col_1 TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl, uid_column='key_1')

    db.insert('table_1', {'key_1': 1, 'col_1': 'value_1'})
    db.insert('table_1', {'key_1': 2, 'col_1': 'value_2'})

    db.update('table_1', {'col_1': 'value_3'}, uid=1)
    results = db.select('SELECT * FROM table_1 WHERE key_1 = :key_1', key_1=1)
    assert results[0]['col_1'] == 'value_3'

    db.delete('table_1', uid=2)
    results = db.select('SELECT * FROM table_1')
    assert len(results) == 1
    assert results[0]['key_1'] == 1


def _mp_worker(n, db_path, num_loops):
    """Worker function for multiprocessing concurrency test."""

    db = sqlitex.Object(str(db_path))

    for i in range(num_loops):
        db.execute(
            """
            UPDATE table_1 SET
                col_1 = col_1 + 1,
                col_2 = :col_2
            WHERE uid = 1
        """,
            col_2=n,
        )


def test_concurrency(tmp_path):
    """Test concurrent writes using multiprocessing (true parallelism)."""
    num_processes = 50
    num_loops = 10
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS table_1 (
            uid INTEGER PRIMARY KEY,
            col_1 INTEGER,
            col_2 INTEGER
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)
    db.insert('table_1', {'uid': 1, 'col_1': 0, 'col_2': -1})

    ps = []

    for n in range(num_processes):
        p = multiprocessing.Process(target=_mp_worker, args=[n, db_path, num_loops])
        ps.append(p)
        p.start()

    for p in ps:
        p.join()

    assert [p.exitcode for p in ps] == [0] * num_processes

    db = sqlitex.Object(str(db_path))

    results = db.select('SELECT col_1 FROM table_1 WHERE uid = 1')
    expected_value = num_processes * num_loops
    assert results[0]['col_1'] == expected_value
