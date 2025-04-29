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

# @TODO other provider methods
