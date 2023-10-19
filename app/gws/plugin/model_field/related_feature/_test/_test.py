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
            fields+ { name "pp" type "text" }
        }
        
        models+ { 
            uid "CHILD" type "postgres" tableName "child"
            fields+ { name "id" type "integer" }
            fields+ { name "cc" type "text" }
            fields+ { 
                name "parent"  
                type relatedFeature 
                fromColumn "parent_k" 
                toModel "PARENT" 
                toColumn "k" 
            }
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

    child = u.model(gws_root, 'CHILD')
    fs = child.find_features(gws.SearchQuery(uids=[2, 3]), mc)

    assert set(f.get('cc') for f in fs) == {'c2', 'c3'}
    assert fs[0].get('parent') is None
    assert fs[1].get('parent') is None


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
        UPDATE child SET parent_k=33 WHERE id=3;
    ''')

    child = u.model(gws_root, 'CHILD')
    fs = child.find_features(gws.SearchQuery(uids=[1, 2, 3, 4]), mc)

    assert set(f.get('cc') for f in fs) == {'c1', 'c2', 'c3', 'c4'}
    assert fs[0].get('parent').get('pp') == 'p11'
    assert fs[1].get('parent').get('pp') == 'p11'
    assert fs[2].get('parent').get('pp') == 'p33'
    assert fs[3].get('parent') is None


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
        UPDATE child SET parent_k=99 WHERE id=4;
    ''')

    parent = u.model(gws_root, 'PARENT')
    child = u.model(gws_root, 'CHILD')

    fs_in = [
        u.feature(child, id=1, parent=u.feature(parent, id=1)),
        u.feature(child, id=2, parent=u.feature(parent, id=5)),
        u.feature(child, id=3, parent=None),
        u.feature(child, id=4, parent=u.feature(parent, id=1)),
        u.feature(child, id=5, parent=u.feature(parent, id=999)),
    ]

    child.update_features(fs_in, mc)

    rows = u.pg_rows('SELECT id, parent_k FROM child WHERE id IN (1,2,3,4,5) ORDER BY id')
    assert rows == [(1, 11), (2, 55), (3, None), (4, 11), (5, None)]


def test_create(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        UPDATE child SET parent_k=0;
    ''')

    parent = u.model(gws_root, 'PARENT')
    child = u.model(gws_root, 'CHILD')

    fs_in = [
        u.feature(child, id=101, parent=u.feature(parent, id=1)),
        u.feature(child, id=102, parent=u.feature(parent, id=5)),
        u.feature(child, id=103, parent=None),
    ]

    child.create_features(fs_in, mc)

    rows = u.pg_rows('SELECT id, parent_k FROM child WHERE id IN (101,102,103) ORDER BY id')
    assert rows == [(101, 11), (102, 55), (103, None)]
