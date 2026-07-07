import gws
import gws.lib.sa as sa
import gws.test.util as u
import gws.base.database.provider


@u.fixture(scope='module')
def db():
    u.pg.create_schema('s1')
    u.pg.create_schema('s2')
    u.pg.create('s1.tab1', {'id': 'int primary key', 'a': 'text'})
    u.pg.create('s2.tab2', {'id': 'int primary key', 'a': 'text'})

    root = u.gws_root()
    yield u.get_db(root)


##


def test_table(db):
    tab = db.table('s1.tab1')
    assert tab is not None
    with u.raises(sa.Error):
        tab = db.table('s1.ZZZ')


def test_count(db: gws.DatabaseProvider):
    tab = db.table('s1.tab1')

    with db.connect() as conn:
        conn.exec_commit('truncate s1.tab1')
        conn.exec_commit(
            tab.insert().values(
            [
                {'id': 1, 'a': 'X'},
                {'id': 2, 'a': 'Y'},
                {'id': 3, 'a': 'Z'},
            ]),
        )
        conn.commit()
        assert db.count(tab) == 3


def test_has_table(db):
    assert db.has_table('s1.tab1') is True
    assert db.has_table('s1.nonexistent') is False


def test_has_schema(db):
    assert db.has_schema('s1') is True
    assert db.has_schema('s2') is True
    assert db.has_schema('no_such_schema') is False


def test_column(db):
    col = db.column('s1.tab1', 'a')
    assert col is not None
    with u.raises(sa.Error):
        db.column('s1.tab1', 'no_such_column')


def test_has_column(db):
    assert db.has_column('s1.tab1', 'id') is True
    assert db.has_column('s1.tab1', 'a') is True
    assert db.has_column('s1.tab1', 'missing') is False


def test_select_text(db):
    with db.connect() as conn:
        conn.exec_commit('truncate s1.tab1')
        conn.exec_commit("insert into s1.tab1 (id, a) values (1, 'hello'), (2, 'world')")

    rows = db.select_text('select id, a from s1.tab1 order by id')
    assert rows == [{'id': 1, 'a': 'hello'}, {'id': 2, 'a': 'world'}]


def test_execute_text(db):
    with db.connect() as conn:
        conn.exec_commit('truncate s1.tab1')

    db.execute_text("insert into s1.tab1 (id, a) values (99, 'z')")
    rows = db.select_text('select id, a from s1.tab1')
    assert rows == [{'id': 99, 'a': 'z'}]
