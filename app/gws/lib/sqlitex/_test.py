"""Tests for the sqlitex module."""

import multiprocessing
import gws.lib.sqlitex as sqlitex


def test_basic_insert_and_select(tmp_path):
    """Test basic insert and select operations."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE users (
            uid INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert a record
    db.insert('users', {'uid': 1, 'name': 'Alice', 'email': 'alice@example.com'})

    # Select and verify
    results = db.select('SELECT * FROM users WHERE uid = :uid', uid=1)
    assert len(results) == 1
    assert results[0]['uid'] == 1
    assert results[0]['name'] == 'Alice'
    assert results[0]['email'] == 'alice@example.com'


def test_update_operation(tmp_path):
    """Test update operation."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE users (
            uid INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert a record
    db.insert('users', {'uid': 1, 'name': 'Bob', 'email': 'bob@example.com'})

    # Update the record
    db.update('users', {'name': 'Robert', 'email': 'robert@example.com'}, uid=1)

    # Verify update
    results = db.select('SELECT * FROM users WHERE uid = :uid', uid=1)
    assert len(results) == 1
    assert results[0]['name'] == 'Robert'
    assert results[0]['email'] == 'robert@example.com'


def test_delete_operation(tmp_path):
    """Test delete operation."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE users (
            uid INTEGER PRIMARY KEY,
            name TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert records
    db.insert('users', {'uid': 1, 'name': 'Charlie'})
    db.insert('users', {'uid': 2, 'name': 'Diana'})

    # Delete one record
    db.delete('users', uid=1)

    # Verify deletion
    results = db.select('SELECT * FROM users')
    assert len(results) == 1
    assert results[0]['uid'] == 2
    assert results[0]['name'] == 'Diana'


def test_execute_statement(tmp_path):
    """Test execute method for DML statements."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE products (
            uid INTEGER PRIMARY KEY,
            name TEXT,
            price REAL
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Use execute for insert
    db.execute('INSERT INTO products (uid, name, price) VALUES (:uid, :name, :price)', uid=1, name='Widget', price=9.99)

    # Verify
    results = db.select('SELECT * FROM products WHERE uid = :uid', uid=1)
    assert len(results) == 1
    assert results[0]['name'] == 'Widget'
    assert results[0]['price'] == 9.99


def test_auto_init_on_missing_table(tmp_path):
    """Test that init_ddl runs automatically when table doesn't exist."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE items (
            uid INTEGER PRIMARY KEY,
            description TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # First query should trigger init_ddl
    db.insert('items', {'uid': 1, 'description': 'Test item'})

    # Verify it worked
    results = db.select('SELECT * FROM items')
    assert len(results) == 1
    assert results[0]['description'] == 'Test item'


def test_multiple_inserts(tmp_path):
    """Test multiple insert operations."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE records (
            uid INTEGER PRIMARY KEY,
            value INTEGER
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert multiple records
    for i in range(1, 6):
        db.insert('records', {'uid': i, 'value': i * 10})

    # Verify all records
    results = db.select('SELECT * FROM records ORDER BY uid')
    assert len(results) == 5
    for i, record in enumerate(results, start=1):
        assert record['uid'] == i
        assert record['value'] == i * 10


def test_select_with_parameters(tmp_path):
    """Test select with various parameter bindings."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE employees (
            uid INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary REAL
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert test data
    db.insert('employees', {'uid': 1, 'name': 'Alice', 'department': 'Engineering', 'salary': 75000})
    db.insert('employees', {'uid': 2, 'name': 'Bob', 'department': 'Engineering', 'salary': 80000})
    db.insert('employees', {'uid': 3, 'name': 'Charlie', 'department': 'Sales', 'salary': 65000})

    # Test filtering by department
    results = db.select('SELECT * FROM employees WHERE department = :dept', dept='Engineering')
    assert len(results) == 2

    # Test filtering by salary range
    results = db.select('SELECT * FROM employees WHERE salary > :min_salary', min_salary=70000)
    assert len(results) == 2


def test_empty_select(tmp_path):
    """Test select that returns no results."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE data (
            uid INTEGER PRIMARY KEY,
            value TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Select from empty table
    results = db.select('SELECT * FROM data')
    assert len(results) == 0
    assert isinstance(results, list)


def test_update_nonexistent_record(tmp_path):
    """Test updating a record that doesn't exist."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE items (
            uid INTEGER PRIMARY KEY,
            name TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Try to update non-existent record (should not raise error)
    db.update('items', {'name': 'Updated'}, uid=999)

    # Verify nothing was changed
    results = db.select('SELECT * FROM items')
    assert len(results) == 0


def test_delete_nonexistent_record(tmp_path):
    """Test deleting a record that doesn't exist."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE items (
            uid INTEGER PRIMARY KEY,
            name TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Try to delete non-existent record (should not raise error)
    db.delete('items', uid=999)

    # Verify table is still empty
    results = db.select('SELECT * FROM items')
    assert len(results) == 0


def test_without_init_ddl(tmp_path):
    """Test creating database without init_ddl."""
    db_path = tmp_path / 'test.db'

    # Create database without init_ddl
    db = sqlitex.Object(str(db_path))

    # Manually create table
    db.execute("""
        CREATE TABLE IF NOT EXISTS simple (
            uid INTEGER PRIMARY KEY,
            data TEXT
        )
    """)

    # Insert and verify
    db.insert('simple', {'uid': 1, 'data': 'test'})
    results = db.select('SELECT * FROM simple')
    assert len(results) == 1


def test_complex_query(tmp_path):
    """Test more complex SQL queries."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE orders (
            uid INTEGER PRIMARY KEY,
            customer TEXT,
            amount REAL,
            status TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert test data
    db.insert('orders', {'uid': 1, 'customer': 'John', 'amount': 100.50, 'status': 'completed'})
    db.insert('orders', {'uid': 2, 'customer': 'Jane', 'amount': 250.75, 'status': 'pending'})
    db.insert('orders', {'uid': 3, 'customer': 'John', 'amount': 75.25, 'status': 'completed'})

    # Complex query with aggregation
    results = db.select(
        """
        SELECT customer, SUM(amount) as total, COUNT(*) as order_count
        FROM orders
        WHERE status = :status
        GROUP BY customer
        ORDER BY total DESC
    """,
        status='completed',
    )

    assert len(results) == 1
    assert results[0]['customer'] == 'John'
    assert results[0]['total'] == 175.75
    assert results[0]['order_count'] == 2


def test_special_characters_in_data(tmp_path):
    """Test handling of special characters in data."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE messages (
            uid INTEGER PRIMARY KEY,
            content TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert data with special characters
    special_text = 'Hello \'world\' with "quotes" and \n newlines \t tabs'
    db.insert('messages', {'uid': 1, 'content': special_text})

    # Verify
    results = db.select('SELECT * FROM messages WHERE uid = :uid', uid=1)
    assert len(results) == 1
    assert results[0]['content'] == special_text


def test_null_values(tmp_path):
    """Test handling of NULL values."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE optional_data (
            uid INTEGER PRIMARY KEY,
            name TEXT,
            optional_field TEXT
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)

    # Insert record with NULL
    db.execute('INSERT INTO optional_data (uid, name, optional_field) VALUES (:uid, :name, :opt)', uid=1, name='Test', opt=None)

    # Verify
    results = db.select('SELECT * FROM optional_data WHERE uid = :uid', uid=1)
    assert len(results) == 1
    assert results[0]['name'] == 'Test'
    assert results[0]['optional_field'] is None


def test_reuse_database(tmp_path):
    """Test that database persists and can be reopened."""
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS persistent (
            uid INTEGER PRIMARY KEY,
            value TEXT
        )
    """

    # First connection
    db1 = sqlitex.Object(str(db_path), init_ddl)
    db1.insert('persistent', {'uid': 1, 'value': 'data1'})

    # Second connection to same database
    db2 = sqlitex.Object(str(db_path), init_ddl)
    results = db2.select('SELECT * FROM persistent')

    assert len(results) == 1
    assert results[0]['value'] == 'data1'


def _mp_worker(n, db_path, num_loops):
    """Worker function for multiprocessing concurrency test."""

    db = sqlitex.Object(str(db_path), connect_args={'timeout': 0.0, 'isolation_level': None})

    for i in range(num_loops):
        db.execute(
            """
            UPDATE counter SET 
                value = value + 1, 
                last_updated_by = :pid 
            WHERE uid = 1
        """,
            pid=n,
        )


def test_concurrency(tmp_path):
    """Test concurrent writes using multiprocessing (true parallelism)."""
    num_processes = 50
    num_loops = 10
    db_path = tmp_path / 'test.db'

    init_ddl = """
        CREATE TABLE IF NOT EXISTS counter (
            uid INTEGER PRIMARY KEY,
            value INTEGER,
            last_updated_by INTEGER
        )
    """

    db = sqlitex.Object(str(db_path), init_ddl)
    db.insert('counter', {'uid': 1, 'value': 0, 'last_updated_by': -1})

    ps = []

    for n in range(num_processes):
        p = multiprocessing.Process(target=_mp_worker, args=[n, db_path, num_loops])
        ps.append(p)
        p.start()

    for p in ps:
        p.join()

    db = sqlitex.Object(str(db_path))

    results = db.select('SELECT value FROM counter WHERE uid = 1')
    expected_value = num_processes * num_loops
    assert results[0]['value'] == expected_value
