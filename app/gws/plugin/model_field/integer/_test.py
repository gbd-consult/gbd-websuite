import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('test_table', {'id': 'int primary key', 'number': 'integer'})

    cfg = '''
        models+ { 
            uid "TEST_MODEL" type "postgres" tableName "test_table"
            fields+ { name "id" type "integer" }
            fields+ { name "number" type "integer" }
        }
    '''

    yield u.gws_root(cfg)


def test_create_integer(root: gws.Root):
    mc = u.model_context()

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1, number=1)
    model.create_feature(f, mc)

    rows = u.pg.rows('SELECT id, number FROM test_table ORDER BY id')
    assert rows == [(1, 1)]


def test_read_integer(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table', [
        {'id': 1, 'number': 1},
        {'id': 2, 'number': 2},
        {'id': 3, 'number': None},
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))
    fs = model.get_features([1, 2, 3], mc)

    assert [isinstance(f.get('number'), int) for f in fs]
    assert [f.get('number') for f in fs] == [1, 2, None]


def test_update_integer(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table', [
        {'id': 1, 'number': 1},
        {'id': 2, 'number': 2},
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1, number=3)
    model.update_feature(f, mc)

    rows = u.pg.rows('SELECT id, number FROM test_table ORDER BY id')
    assert rows == [
        (1, 3), (2, 2)
    ]


def test_delete_number(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table', [
        {'id': 1, 'number': 1},
        {'id': 2, 'number': 2},
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1)
    model.delete_feature(f, mc)

    rows = u.pg.rows('SELECT id, number FROM test_table ORDER BY id')
    assert rows == [(2, 2)]
