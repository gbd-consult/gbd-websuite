import gws
import gws.lib.test.util as u


@u.fixture(scope='module')
def gws_root():
    u.pg_create_table(
        'parent',
        {'id': 'int primary key', 'k': 'int', 'pp': 'text'},
        {'id': 1, 'k': 11, 'pp': 'p11'},
        {'id': 2, 'k': 22, 'pp': 'p22'},
        {'id': 3, 'k': 33, 'pp': 'p33'},
        {'id': 4, 'k': 44, 'pp': 'p44'},
        {'id': 5, 'k': 55, 'pp': 'p55'},
    )
    u.pg_create_table(
        'child',
        {'id': 'int primary key', 'cc': 'text', 'parent_k': 'int'},
        {'id': 1, 'cc': 'c1', 'parent_k': 0},
        {'id': 2, 'cc': 'c2', 'parent_k': 0},
        {'id': 3, 'cc': 'c3', 'parent_k': 0},
        {'id': 4, 'cc': 'c4', 'parent_k': 0},
        {'id': 5, 'cc': 'c5', 'parent_k': 0},
        {'id': 6, 'cc': 'c6', 'parent_k': 0},
        {'id': 7, 'cc': 'c7', 'parent_k': 0},
        {'id': 8, 'cc': 'c8', 'parent_k': 0},
        {'id': 9, 'cc': 'c9', 'parent_k': 0},
    )

    cfg = '''
        database.providers+ { type "postgres" serviceName "gws_test_postgres" } 
        
        models+ { 
            uid "PARENT" type "postgres" tableName "parent"
            fields+ { name "id" type "integer" }
            fields+ { name "k" type "text" }
            fields+ { name "pp" type "text" }
            fields+ { 
                name "children"  
                type relatedFeatureList 
                fromColumn "k" toModel "CHILD" toColumn "parent_k"
            }
        }
        
        models+ { 
            uid "CHILD" type "postgres" tableName "child"
            fields+ { name "id" type "integer" }
            fields+ { name "cc" type "text" }
        }
    '''

    yield u.gws_configure(cfg)


def test_find_no_depth(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
    )
    u.pg_exec('''
        UPDATE child SET parent_k=11;
    ''')

    parent = u.db_model(gws_root, 'PARENT')
    fs = parent.find_features(gws.SearchQuery(uids=[1, 3, 5]), mc)

    assert set(f.get('pp') for f in fs) == {'p11', 'p33', 'p55'}
    assert fs[0].get('children') is None
    assert fs[1].get('children') is None
    assert fs[2].get('children') is None


def test_find_depth(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        UPDATE child SET parent_k=0;
        UPDATE child SET parent_k=11 WHERE id=1;
        UPDATE child SET parent_k=11 WHERE id=2;
        UPDATE child SET parent_k=11 WHERE id=3;
        UPDATE child SET parent_k=33 WHERE id=4;
    ''')

    parent = u.db_model(gws_root, 'PARENT')
    fs = parent.find_features(gws.SearchQuery(uids=[1, 3, 5]), mc)

    assert set(f.get('pp') for f in fs) == {'p11', 'p33', 'p55'}

    assert set(c.get('cc') for c in fs[0].get('children')) == {'c1', 'c2', 'c3'}
    assert set(c.get('cc') for c in fs[1].get('children')) == {'c4'}
    assert set(c.get('cc') for c in fs[2].get('children')) == set()


def test_update(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        UPDATE child SET parent_k=0;
        UPDATE child SET parent_k=11 WHERE id=1;
        UPDATE child SET parent_k=11 WHERE id=2;
        UPDATE child SET parent_k=33 WHERE id=3;
    ''')

    parent = u.db_model(gws_root, 'PARENT')
    child = u.db_model(gws_root, 'CHILD')

    fs_in = [
        u.feature(parent, id=1),
        u.feature(parent, id=2),
    ]
    fs_in[0].set('children', [
        u.feature(child, id=1),
        u.feature(child, id=2),
    ])
    fs_in[1].set('children', [
        u.feature(child, id=3),
        u.feature(child, id=4),
    ])

    parent.update_features(fs_in, mc)

    rows = u.pg_rows('SELECT id, parent_k FROM child WHERE id IN (1,2,3,4) ORDER BY id')
    assert rows == [(1, 11), (2, 11), (3, 22), (4, 22)]


def test_create(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        UPDATE child SET parent_k=0;
        UPDATE child SET parent_k=11 WHERE id=1;
        UPDATE child SET parent_k=11 WHERE id=2;
        UPDATE child SET parent_k=33 WHERE id=3;
    ''')

    parent = u.db_model(gws_root, 'PARENT')
    child = u.db_model(gws_root, 'CHILD')

    fs_in = [
        u.feature(parent, id=101, k='111'),
        u.feature(parent, id=102, k='222'),
        u.feature(parent, id=103, k='333'),
    ]
    fs_in[0].set('children', [
        u.feature(child, id=1),
        u.feature(child, id=2),
        u.feature(child, id=3),
    ])
    fs_in[1].set('children', [
        u.feature(child, id=4),
    ])

    parent.create_features(fs_in, mc)

    rows = u.pg_rows('SELECT id, parent_k FROM child WHERE id IN (1,2,3,4) ORDER BY id')
    assert rows == [(1, 111), (2, 111), (3, 111), (4, 222)]


def test_delete(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        UPDATE child SET parent_k=0;
        UPDATE child SET parent_k=11 WHERE id=1;
        UPDATE child SET parent_k=11 WHERE id=2;
        UPDATE child SET parent_k=33 WHERE id=3;
        UPDATE child SET parent_k=33 WHERE id=4;
    ''')

    parent = u.db_model(gws_root, 'PARENT')

    fs_in = [
        u.feature(parent, id=1),
        u.feature(parent, id=3),
    ]

    parent.delete_features(fs_in, mc)

    rows = u.pg_rows('SELECT id, parent_k FROM child WHERE id IN (1,2,3,4) ORDER BY id')
    assert rows == [(1, None), (2, None), (3, None), (4, None)]
