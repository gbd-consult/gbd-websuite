import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('test_table', {'id': 'int primary key', 'bool_field': 'boolean'})

    cfg = '''
        models+ { 
            uid "TEST_MODEL" type "postgres" tableName "test_table"
            fields+ { name "id" type "integer" }
            fields+ { name "bool_field" type "bool" }
        }
    '''

    yield u.gws_root(cfg)


def test_create_boolean(root: gws.Root):
    mc = u.model_context()

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1, bool_field=True)
    model.create_feature(f, mc)

    rows = u.pg.rows('SELECT id, bool_field FROM test_table ORDER BY id')
    assert rows == [(1, True)]


def test_read_boolean(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table', [
        {'id': 1, 'bool_field': False},
        {'id': 2, 'bool_field': True},
        {'id': 3, 'bool_field': None},
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))
    fs = model.get_features([1, 2, 3], mc)

    assert [f.get('bool_field') for f in fs] == [False, True, None]


def test_update_boolean(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table', [
        {'id': 1, 'bool_field': True},
        {'id': 2, 'bool_field': False},
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1, bool_field=False)
    model.update_feature(f, mc)

    rows = u.pg.rows('SELECT id, bool_field FROM test_table ORDER BY id')
    assert rows == [
        (1, False), (2, False)
    ]


def test_delete_boolean(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table', [
        {'id': 1, 'bool_field': True},
        {'id': 2, 'bool_field': False},
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1)
    model.delete_feature(f, mc)

    rows = u.pg.rows('SELECT id, bool_field FROM test_table ORDER BY id')
    assert rows == [(2, False)]
