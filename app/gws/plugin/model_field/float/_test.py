import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('test_table', {'id': 'int primary key', 'number': 'float'})

    cfg = '''
        models+ { 
            uid "TEST_MODEL" type "postgres" tableName "test_table"
            fields+ { name "id" type "integer" }
            fields+ { name "number" type "float" }
        }
    '''

    yield u.gws_root(cfg)


def test_create_float(root: gws.Root):
    mc = u.model_context()

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1, number=1.5)
    model.create_feature(f, mc)

    rows = u.pg.rows('SELECT id, number FROM test_table ORDER BY id')
    assert rows == [(1, 1.5)]


def test_read_float(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table', [
        {'id': 1, 'number': 1.5},
        {'id': 2, 'number': 2.5},
        {'id': 3, 'number': None},
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))
    fs = model.get_features([1, 2, 3], mc)

    assert [isinstance(f.get('number'),float) for f in fs]
    assert [f.get('number') for f in fs] == [1.5, 2.5, None]


def test_update_float(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table', [
        {'id': 1, 'number': 1.5},
        {'id': 2, 'number': 2.5},
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1, number=3.5)
    model.update_feature(f, mc)

    rows = u.pg.rows('SELECT id, number FROM test_table ORDER BY id')
    assert rows == [
        (1, 3.5), (2, 2.5)
    ]


def test_delete_number(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table', [
        {'id': 1, 'number': 1.5},
        {'id': 2, 'number': 2.5},
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1)
    model.delete_feature(f, mc)

    rows = u.pg.rows('SELECT id, number FROM test_table ORDER BY id')
    assert rows == [(2, 2.5)]
