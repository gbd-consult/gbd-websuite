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


def test_fetch_all(db):
    with db.connect() as conn:
        conn.exec_commit("truncate tab")
        conn.exec_commit("insert into tab (id, a) values (1, 'X'), (2, 'Y')")

    with db.connect() as conn:
        rows = conn.fetch_all("select id, a from tab order by id")
    assert rows == [{'id': 1, 'a': 'X'}, {'id': 2, 'a': 'Y'}]


def test_fetch_first(db):
    with db.connect() as conn:
        conn.exec_commit("truncate tab")
        conn.exec_commit("insert into tab (id, a) values (1, 'X'), (2, 'Y')")

    with db.connect() as conn:
        row = conn.fetch_first("select id, a from tab order by id")
    assert row == {'id': 1, 'a': 'X'}


def test_fetch_first_empty(db):
    with db.connect() as conn:
        conn.exec_commit("truncate tab")

    with db.connect() as conn:
        row = conn.fetch_first("select id, a from tab")
    assert row is None


def test_fetch_scalars(db):
    with db.connect() as conn:
        conn.exec_commit("truncate tab")
        conn.exec_commit("insert into tab (id, a) values (10, 'A'), (20, 'B'), (30, 'C')")

    with db.connect() as conn:
        vals = conn.fetch_scalars("select id from tab order by id")
    assert vals == [10, 20, 30]


def test_fetch_ints(db):
    with db.connect() as conn:
        conn.exec_commit("truncate tab")
        conn.exec_commit("insert into tab (id, a) values (7, 'A'), (8, 'B')")

    with db.connect() as conn:
        vals = conn.fetch_ints("select id from tab order by id")
    assert vals == [7, 8]


def test_fetch_strings(db):
    with db.connect() as conn:
        conn.exec_commit("truncate tab")
        conn.exec_commit("insert into tab (id, a) values (1, 'hello'), (2, 'world')")

    with db.connect() as conn:
        vals = conn.fetch_strings("select a from tab order by id")
    assert vals == ['hello', 'world']


def test_fetch_scalar(db):
    with db.connect() as conn:
        val = conn.fetch_scalar("select 42")
    assert val == 42


def test_fetch_string(db):
    with db.connect() as conn:
        val = conn.fetch_string("select 'hello'")
    assert val == 'hello'


def test_fetch_int(db):
    with db.connect() as conn:
        val = conn.fetch_int("select 99")
    assert val == 99
