import gws
import gws.test.util as u
import gws.lib.datetimex


@u.fixture(scope='module')
def root():
    u.pg.create('test_table', {'id': 'int primary key', 'datetime': 'text'})

    cfg = '''
        models+ { 
            uid "TEST_MODEL" type "postgres" tableName "test_table"
            fields+ { name "id" type "integer" }
            fields+ { name "datetime" type "text" }
        }
    '''

    yield u.gws_root(cfg)


def test_create_date(root: gws.Root):
    mc = u.model_context()

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    # print(gws.lib.datetimex.new(2010,6,18)) #2010-06-18 00:00:00+02:00

    d = gws.lib.datetimex.new(2010, 6, 18)
    f = u.feature(model, id=1, datetime=d)
    model.create_feature(f, mc)

    iso = gws.lib.datetimex.to_iso_date_string(d)

    rows = u.pg.rows('SELECT id, datetime FROM test_table ORDER BY id')
    assert rows == [(1, iso)]

def test_read_date(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table',[
        {'id':1, 'datetime':'2010-06-18 00:00:00+00'},
        {'id':2, 'datetime':'2020-07-20 00:00:00+00'}
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))
    fs = model.get_features([1,2], mc)

    assert [gws.lib.datetimex.is_datetime(f) for f in fs]
    assert [f.get('datetime') for f in fs] == ['2010-06-18 00:00:00+00', '2020-07-20 00:00:00+00']

def test_update_date(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table',[
        {'id':1, 'datetime':'2010-06-18 00:00:00+00'},
        {'id':2, 'datetime':'2020-07-20 00:00:00+00'}
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1, datetime='2010-06-19 00:00:00+00')
    model.update_feature(f,mc)

    rows = u.pg.rows('SELECT id, datetime FROM test_table ORDER BY datetime')
    assert rows == [(1,'2010-06-19 00:00:00+00'),(2,'2020-07-20 00:00:00+00')]


def test_delete_date(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('test_table',[
        {'id':1, 'datetime':'2010-06-18 00:00:00+00'},
        {'id':2, 'datetime':'2020-07-20 00:00:00+00'}
    ])

    model = u.cast(gws.Model, root.get('TEST_MODEL'))

    f = u.feature(model, id=1)
    model.delete_feature(f,mc)

    rows = u.pg.rows('SELECT id, datetime FROM test_table ORDER BY id')
    assert rows == [(2,'2020-07-20 00:00:00+00')]
