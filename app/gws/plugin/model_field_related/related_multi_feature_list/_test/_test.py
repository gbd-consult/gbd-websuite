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
        'a',
        {'id': 'int primary key', 'cc': 'text', 'parent_k': 'int'},
        {'id': 1, 'cc': 'a1', 'parent_k': 0},
        {'id': 2, 'cc': 'a2', 'parent_k': 0},
        {'id': 3, 'cc': 'a3', 'parent_k': 0},
        {'id': 4, 'cc': 'a4', 'parent_k': 0},
        {'id': 5, 'cc': 'a5', 'parent_k': 0},
        {'id': 6, 'cc': 'a6', 'parent_k': 0},
    )
    u.pg_create_table(
        'b',
        {'id': 'int primary key', 'cc': 'text', 'parent_k': 'int'},
        {'id': 1, 'cc': 'b1', 'parent_k': 0},
        {'id': 2, 'cc': 'b2', 'parent_k': 0},
        {'id': 3, 'cc': 'b3', 'parent_k': 0},
        {'id': 4, 'cc': 'b4', 'parent_k': 0},
        {'id': 5, 'cc': 'b5', 'parent_k': 0},
        {'id': 6, 'cc': 'b6', 'parent_k': 0},
    )
    u.pg_create_table(
        'c',
        {'id': 'int primary key', 'cc': 'text', 'parent_k': 'int'},
        {'id': 1, 'cc': 'c1', 'parent_k': 0},
        {'id': 2, 'cc': 'c2', 'parent_k': 0},
        {'id': 3, 'cc': 'c3', 'parent_k': 0},
        {'id': 4, 'cc': 'c4', 'parent_k': 0},
        {'id': 5, 'cc': 'c5', 'parent_k': 0},
        {'id': 6, 'cc': 'c6', 'parent_k': 0},
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
                type relatedMultiFeatureList
                fromColumn "k" 
                related+ { toModel "A" toColumn "parent_k" }
                related+ { toModel "B" toColumn "parent_k" }
                related+ { toModel "C" toColumn "parent_k" }
            }
        }
        
        models+ { uid "A" type "postgres" tableName "a" fields+ { name "id" type "integer" } fields+ { name "cc" type "text" } }
        models+ { uid "B" type "postgres" tableName "b" fields+ { name "id" type "integer" } fields+ { name "cc" type "text" } }
        models+ { uid "C" type "postgres" tableName "c" fields+ { name "id" type "integer" } fields+ { name "cc" type "text" } }
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
        UPDATE a SET parent_k=0;
        UPDATE b SET parent_k=0;
        UPDATE c SET parent_k=0;
        
        UPDATE a SET parent_k=11 WHERE id=1;
        UPDATE a SET parent_k=11 WHERE id=2;
        UPDATE a SET parent_k=11 WHERE id=3;
        UPDATE a SET parent_k=33 WHERE id=4;

        UPDATE b SET parent_k=11 WHERE id=4;
        UPDATE b SET parent_k=11 WHERE id=5;

        UPDATE c SET parent_k=33 WHERE id=6;
    ''')

    parent = u.db_model(gws_root, 'PARENT')
    fs = parent.find_features(gws.SearchQuery(uids=[1, 3, 5]), mc)

    assert set(f.get('pp') for f in fs) == {'p11', 'p33', 'p55'}

    assert set(c.get('cc') for c in fs[0].get('children')) == {'a1', 'a2', 'a3', 'b4', 'b5'}
    assert set(c.get('cc') for c in fs[1].get('children')) == {'a4', 'c6'}
    assert set(c.get('cc') for c in fs[2].get('children')) == set()


_SELECT_ALL = '''
    SELECT p.parent_k, p.t, p.id FROM (
        (SELECT 'a' AS t, id, parent_k FROM a)
        UNION (SELECT 'b' AS t, id, parent_k FROM b)
        UNION (SELECT 'c' AS t, id, parent_k FROM c)
    ) AS p WHERE p.parent_k IS NULL OR p.parent_k > 0
    ORDER BY p.t, p.id 
'''


def test_update(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        UPDATE a SET parent_k=0;
        UPDATE b SET parent_k=0;
        UPDATE c SET parent_k=0;
        
        UPDATE a SET parent_k=11 WHERE id=1;
        UPDATE a SET parent_k=11 WHERE id=2;
        UPDATE a SET parent_k=11 WHERE id=3;
        UPDATE a SET parent_k=11 WHERE id=4;
        UPDATE b SET parent_k=11 WHERE id=4;
        UPDATE b SET parent_k=11 WHERE id=5;
        
        UPDATE a SET parent_k=22 WHERE id=5;

        UPDATE a SET parent_k=33 WHERE id=6;
        UPDATE c SET parent_k=33 WHERE id=6;
    ''')

    parent = u.db_model(gws_root, 'PARENT')
    a = u.db_model(gws_root, 'A')
    b = u.db_model(gws_root, 'B')
    c = u.db_model(gws_root, 'C')

    fs_in = [
        u.feature(parent, id=1),
        u.feature(parent, id=2),
    ]
    fs_in[0].set('children', [
        u.feature(a, id=1), u.feature(a, id=2),
        u.feature(b, id=4), u.feature(b, id=6),
        u.feature(c, id=1), u.feature(c, id=6),
    ])
    fs_in[1].set('children', [
        u.feature(a, id=3),
        u.feature(b, id=4),
    ])

    parent.update_features(fs_in, mc)

    assert u.pg_rows(_SELECT_ALL) == [
        (11, 'a', 1),
        (11, 'a', 2),
        (22, 'a', 3),
        (None, 'a', 4),
        (None, 'a', 5),
        (33, 'a', 6),
        (22, 'b', 4),
        (None, 'b', 5),
        (11, 'b', 6),
        (11, 'c', 1),
        (11, 'c', 6),
    ]


def test_create(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        UPDATE a SET parent_k=0;
        UPDATE b SET parent_k=0;
        UPDATE c SET parent_k=0;
        
        UPDATE a SET parent_k=11 WHERE id=1;
        UPDATE a SET parent_k=11 WHERE id=2;
        UPDATE a SET parent_k=11 WHERE id=3;
        UPDATE a SET parent_k=11 WHERE id=4;
        UPDATE b SET parent_k=11 WHERE id=4;
        UPDATE b SET parent_k=11 WHERE id=5;
        
        UPDATE a SET parent_k=22 WHERE id=5;

        UPDATE a SET parent_k=33 WHERE id=6;
        UPDATE c SET parent_k=33 WHERE id=6;
    ''')

    parent = u.db_model(gws_root, 'PARENT')
    a = u.db_model(gws_root, 'A')
    b = u.db_model(gws_root, 'B')
    c = u.db_model(gws_root, 'C')

    fs_in = [
        u.feature(parent, id=101, k='111'),
        u.feature(parent, id=102, k='222'),
        u.feature(parent, id=103, k='333'),
    ]
    fs_in[0].set('children', [
        u.feature(a, id=1), u.feature(a, id=2), u.feature(a, id=4),
        u.feature(b, id=2),
        u.feature(c, id=6),
    ])
    fs_in[1].set('children', [
        u.feature(a, id=3),
        u.feature(c, id=3),
    ])
    fs_in[2].set('children', [

    ])

    parent.create_features(fs_in, mc)

    assert u.pg_rows(_SELECT_ALL) == [
        (111, 'a', 1),
        (111, 'a', 2),
        (222, 'a', 3),
        (111, 'a', 4),
        (22, 'a', 5),
        (33, 'a', 6),
        (111, 'b', 2),
        (11, 'b', 4),
        (11, 'b', 5),
        (222, 'c', 3),
        (111, 'c', 6),
    ]


def test_delete(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        UPDATE a SET parent_k=0;
        UPDATE b SET parent_k=0;
        UPDATE c SET parent_k=0;
        
        UPDATE a SET parent_k=11 WHERE id=1;
        UPDATE a SET parent_k=11 WHERE id=2;
        UPDATE a SET parent_k=11 WHERE id=3;
        UPDATE a SET parent_k=11 WHERE id=4;
        UPDATE b SET parent_k=11 WHERE id=4;
        UPDATE b SET parent_k=11 WHERE id=5;
        
        UPDATE a SET parent_k=22 WHERE id=5;

        UPDATE a SET parent_k=33 WHERE id=6;
        UPDATE c SET parent_k=33 WHERE id=6;
    ''')

    parent = u.db_model(gws_root, 'PARENT')

    fs_in = [
        u.feature(parent, id=1),
        u.feature(parent, id=3),
    ]

    parent.delete_features(fs_in, mc)

    assert u.pg_rows(_SELECT_ALL) == [
        (None, 'a', 1),
        (None, 'a', 2),
        (None, 'a', 3),
        (None, 'a', 4),
        (22, 'a', 5),
        (None, 'a', 6),
        (None, 'b', 4),
        (None, 'b', 5),
        (None, 'c', 6),
    ]
