import gws
import gws.lib.sa as sa
import gws.test.util as u
import gws.test.util.options as options


@u.fixture(scope='module')
def root():
    u.pg.create('tab', {'id': 'int primary key', 'a': 'text'})
    cfg = f"""
        database.providers+ {{
            uid "provider_2"
            type "postgres"
            host     {options.option('service.postgres.host')!r}
            port     {int(options.option('service.postgres.port'))}
            username {options.option('service.postgres.user')!r}
            password {options.option('service.postgres.password')!r}
            database {options.option('service.postgres.database')!r}
        }}
    """
    yield u.gws_root(cfg)


@u.fixture(scope='module')
def db(root):
    yield u.pg.provider(root)


@u.fixture(scope='module')
def db_2(root):
    yield root.get('provider_2')


def _count(db):
    with db.begin() as conn:
        return conn.fetch_int('select count(*) from tab')


##


def test_begin(db):
    with db.begin() as conn:
        assert conn.fetch_int('select 123') == 123


def test_connection_closed(db):
    with db.begin() as conn:
        conn.fetch_int('select 123')
    assert db._sa_connection() is None


def test_nested_begin(db):
    with db.begin() as conn:
        assert conn.fetch_int('select 123') == 123
        assert db._sa_connection() is not None
        with db.begin() as conn:
            assert conn.fetch_int('select 234') == 234
        assert db._sa_connection() is not None

    assert db._sa_connection() is None


def test_commit(db):
    with db.begin() as conn:
        conn.execute("insert into tab (id, a) values (1, 'X')")
        assert conn.fetch_int("select id from tab where a = 'X'") == 1

    with db.begin() as conn:
        conn.execute("insert into tab (id, a) values (2, 'Y')")

    with db.begin() as conn:
        assert conn.fetch_int("select id from tab where a = 'Y'") == 2


def test_error_rollback(db):
    with db.begin() as conn:
        cnt_1 = conn.fetch_int("select count(*) from tab")

    with u.raises(Exception):
        with db.begin() as conn:
            conn.execute("insert into tab (id, a) values (100, 'X')")
            conn.execute("insert into tab (id, a) values (NULL, 'X')")

    with db.begin() as conn:
        cnt_2 = conn.fetch_int("select count(*) from tab")

    assert cnt_1 == cnt_2


def test_fetch_all(db):
    with db.begin() as conn:
        conn.execute("truncate tab")
        conn.execute("insert into tab (id, a) values (1, 'X'), (2, 'Y')")

    with db.begin() as conn:
        rows = conn.fetch_all("select id, a from tab order by id")
    assert rows == [{'id': 1, 'a': 'X'}, {'id': 2, 'a': 'Y'}]


def test_fetch_first(db):
    with db.begin() as conn:
        conn.execute("truncate tab")
        conn.execute("insert into tab (id, a) values (1, 'X'), (2, 'Y')")

    with db.begin() as conn:
        row = conn.fetch_first("select id, a from tab order by id")
    assert row == {'id': 1, 'a': 'X'}


def test_fetch_first_empty(db):
    with db.begin() as conn:
        conn.execute("truncate tab")

    with db.begin() as conn:
        row = conn.fetch_first("select id, a from tab")
    assert row is None


def test_fetch_scalars(db):
    with db.begin() as conn:
        conn.execute("truncate tab")
        conn.execute("insert into tab (id, a) values (10, 'A'), (20, 'B'), (30, 'C')")

    with db.begin() as conn:
        vals = conn.fetch_scalars("select id from tab order by id")
    assert vals == [10, 20, 30]


def test_fetch_ints(db):
    with db.begin() as conn:
        conn.execute("truncate tab")
        conn.execute("insert into tab (id, a) values (7, 'A'), (8, 'B')")

    with db.begin() as conn:
        vals = conn.fetch_ints("select id from tab order by id")
    assert vals == [7, 8]


def test_fetch_strings(db):
    with db.begin() as conn:
        conn.execute("truncate tab")
        conn.execute("insert into tab (id, a) values (1, 'hello'), (2, 'world')")

    with db.begin() as conn:
        vals = conn.fetch_strings("select a from tab order by id")
    assert vals == ['hello', 'world']


def test_fetch_scalar(db):
    with db.begin() as conn:
        val = conn.fetch_scalar("select 42")
    assert val == 42


def test_fetch_string(db):
    with db.begin() as conn:
        val = conn.fetch_string("select 'hello'")
    assert val == 'hello'


def test_fetch_int(db):
    with db.begin() as conn:
        val = conn.fetch_int("select 99")
    assert val == 99


def test_fetch_inside_write_keeps_writes(db):
    with db.begin() as conn:
        conn.execute('truncate tab')

    with db.begin() as conn:
        conn.execute("insert into tab (id, a) values (1, 'X')")
        assert conn.fetch_int('select count(*) from tab') == 1
        with db.begin() as conn_2:
            assert conn_2.fetch_all('select id from tab') == [{'id': 1}]

    assert _count(db) == 1


def test_nested_error_rolls_back_all(db):
    with db.begin() as conn:
        conn.execute('truncate tab')

    with u.raises(ValueError):
        with db.begin() as conn:
            conn.execute("insert into tab (id, a) values (1, 'X')")
            with db.begin() as conn_2:
                conn_2.execute("insert into tab (id, a) values (2, 'Y')")
                raise ValueError()

    assert _count(db) == 0


def test_savepoint_error_rolls_back_savepoint(db):
    with db.begin() as conn:
        conn.execute('truncate tab')

    with db.begin() as conn:
        conn.execute("insert into tab (id, a) values (1, 'X')")
        try:
            with db.begin(nested=True) as conn_2:
                conn_2.execute("insert into tab (id, a) values (2, 'Y')")
                conn_2.execute("insert into tab (id, a) values (NULL, 'Z')")
        except sa.Error:
            pass
        conn.execute("insert into tab (id, a) values (3, 'Z')")

    with db.begin() as conn:
        assert conn.fetch_ints('select id from tab order by id') == [1, 3]


def test_savepoint_commits_with_outer(db):
    with db.begin() as conn:
        conn.execute('truncate tab')

    with db.begin() as conn:
        with db.begin(nested=True) as conn_2:
            conn_2.execute("insert into tab (id, a) values (1, 'X')")

    assert _count(db) == 1


def test_connections_are_per_provider(db, db_2):
    with db.begin() as conn:
        conn.execute('truncate tab')

    with db.begin() as conn:
        conn.execute("insert into tab (id, a) values (1, 'X')")
        with db_2.begin() as conn_2:
            assert conn_2.saConn is not conn.saConn
            assert conn_2.fetch_int('select count(*) from tab') == 0
        assert db._sa_connection() is conn.saConn

    assert _count(db_2) == 1


def test_autocommit_connection(db):
    with db.begin() as conn:
        conn.execute('truncate tab')

    with db.autocommit_connection() as conn:
        conn.execute("insert into tab (id, a) values (1, 'X')")
        assert db._sa_connection() is None

    assert _count(db) == 1


def test_autocommit_connection_inside_transaction(db):
    with db.begin():
        with u.raises(gws.Error):
            with db.autocommit_connection():
                pass
