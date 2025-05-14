import gws.test.util as u


@u.fixture(scope='module')
def db():
    u.pg.create('tab', {'id': 'int primary key', 'a': 'text'})
    root = u.gws_root()
    yield u.get_db(root)


##


def test_connect(db):
    with db.connect() as conn:
        assert conn.fetch_int('select 123') == 123


def test_connection_closed(db):
    with db.connect() as conn:
        conn.fetch_int('select 123')
    assert db._sa_connection() is None


def test_nested_connection(db):
    with db.connect() as conn:
        assert conn.fetch_int('select 123') == 123
        assert db._sa_connection() is not None
        with db.connect() as conn:
            assert conn.fetch_int('select 234') == 234
        assert db._sa_connection() is not None

    assert db._sa_connection() is None


def test_commit(db):
    with db.connect() as conn:
        conn.exec_commit("insert into tab (id, a) values (1, 'X')")
        assert conn.fetch_int("select id from tab where a = 'X'") == 1

    with db.connect() as conn:
        conn.exec("insert into tab (id, a) values (2, 'Y')")
        conn.commit()
    
    with db.connect() as conn:
        assert conn.fetch_int("select id from tab where a = 'Y'") == 2


def test_error_rollback(db):
    with db.connect() as conn:
        cnt_1 = conn.fetch_int("select count(*) from tab")

    with u.raises(Exception):
        with db.connect() as conn:
            conn.exec("insert into tab (id, a) values (100, 'X')")
            conn.exec("insert into tab (id, a) values (NULL, 'X')")

    with db.connect() as conn:
        cnt_2 = conn.fetch_int("select count(*) from tab")

    assert cnt_1 == cnt_2

# @TODO other connection methods
