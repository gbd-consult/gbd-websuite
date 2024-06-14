import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('parent', {'id': 'int primary key', 'k': 'int', 'pp': 'text'})
    u.pg.create('parent_auto', {'id': 'int primary key generated always as identity', 'k': 'int', 'pp': 'text'})

    u.pg.create('a', {'id': 'int primary key', 'cc': 'text', 'parent_k': 'int'})
    u.pg.create('b', {'id': 'int primary key', 'cc': 'text', 'parent_k': 'int'})
    u.pg.create('c', {'id': 'int primary key', 'cc': 'text', 'parent_k': 'int'})

    cfg = '''
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

        models+ { 
            uid "PARENT_AUTO" type "postgres" tableName "parent_auto"
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

    yield u.gws_root(cfg)


def test_find_no_depth(root: gws.Root):
    mc = u.model_context()

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    fs = parent.get_features([1, 2], mc)

    assert set(f.get('pp') for f in fs) == {'p1', 'p2'}
    assert fs[0].get('children') is None
    assert fs[1].get('children') is None


def test_find_depth(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
        {'id': 3, 'k': 33, 'pp': 'p3'},
    ])
    u.pg.insert('a', [
        {'id': 1, 'cc': 'a1', 'parent_k': 11},
        {'id': 2, 'cc': 'a2', 'parent_k': 22},
        {'id': 3, 'cc': 'a3', 'parent_k': 99},
    ])
    u.pg.insert('b', [
        {'id': 1, 'cc': 'b1', 'parent_k': 11},
        {'id': 2, 'cc': 'b2', 'parent_k': 11},
    ])
    u.pg.insert('c', [
        {'id': 1, 'cc': 'c1', 'parent_k': 22},
        {'id': 2, 'cc': 'c2', 'parent_k': 99},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    fs = parent.get_features([1, 2, 3], mc)

    assert [f.get('pp') for f in fs] == ['p1', 'p2', 'p3']

    assert [c.get('cc') for c in fs[0].get('children')] == ['a1', 'b1', 'b2']
    assert [c.get('cc') for c in fs[1].get('children')] == ['a2', 'c1']
    assert [c.get('cc') for c in fs[2].get('children')] == []


_SELECT_ALL = '''
    SELECT p.t, p.id, p.parent_k FROM (
        (SELECT 'a' AS t, id, parent_k FROM a)
        UNION 
        (SELECT 'b' AS t, id, parent_k FROM b)
        UNION 
        (SELECT 'c' AS t, id, parent_k FROM c)
    ) AS p
    ORDER BY p.t, p.id, p.parent_k
'''


def test_update(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
        {'id': 3, 'k': 33, 'pp': 'p3'},
    ])
    u.pg.insert('a', [
        {'id': 1, 'cc': 'a1', 'parent_k': 11},
        {'id': 2, 'cc': 'a2', 'parent_k': 11},
        {'id': 3, 'cc': 'a3', 'parent_k': 99},
    ])
    u.pg.insert('b', [
        {'id': 3, 'cc': 'b3', 'parent_k': 11},
        {'id': 4, 'cc': 'b4', 'parent_k': 11},
    ])
    u.pg.insert('c', [
        {'id': 5, 'cc': 'c5', 'parent_k': 11},
        {'id': 6, 'cc': 'c6', 'parent_k': 11},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    a = u.cast(gws.Model, root.get('A'))
    b = u.cast(gws.Model, root.get('B'))
    c = u.cast(gws.Model, root.get('C'))

    f = u.feature(parent, id=1, children=[
        u.feature(a, id=1),
        u.feature(a, id=3),
        u.feature(b, id=4),
    ])

    parent.update_feature(f, mc)

    assert u.pg.rows(_SELECT_ALL) == [
        ('a', 1, 11),
        ('a', 2, None),
        ('a', 3, 11),
        ('b', 3, None),
        ('b', 4, 11),
        ('c', 5, None),
        ('c', 6, None),
    ]


def test_create(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
        {'id': 3, 'k': 33, 'pp': 'p3'},
    ])
    u.pg.insert('a', [
        {'id': 1, 'cc': 'a1', 'parent_k': 11},
        {'id': 2, 'cc': 'a2', 'parent_k': 11},
        {'id': 3, 'cc': 'a2', 'parent_k': 11},
    ])
    u.pg.insert('b', [
        {'id': 3, 'cc': 'b3', 'parent_k': 11},
        {'id': 4, 'cc': 'b4', 'parent_k': 11},
    ])
    u.pg.insert('c', [
        {'id': 5, 'cc': 'c5', 'parent_k': 11},
        {'id': 6, 'cc': 'c6', 'parent_k': 11},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    a = u.cast(gws.Model, root.get('A'))
    b = u.cast(gws.Model, root.get('B'))
    c = u.cast(gws.Model, root.get('C'))

    f = u.feature(parent, id=999, k=99, children=[
        u.feature(a, id=1),
        u.feature(a, id=3),
        u.feature(b, id=4),
    ])

    parent.create_feature(f, mc)

    assert u.pg.rows(_SELECT_ALL) == [
        ('a', 1, 99),
        ('a', 2, 11),
        ('a', 3, 99),
        ('b', 3, 11),
        ('b', 4, 99),
        ('c', 5, 11),
        ('c', 6, 11),
    ]


def test_create_auto(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
        {'id': 3, 'k': 33, 'pp': 'p3'},
    ])
    u.pg.insert('a', [
        {'id': 1, 'cc': 'a1', 'parent_k': 11},
        {'id': 2, 'cc': 'a2', 'parent_k': 11},
        {'id': 3, 'cc': 'a2', 'parent_k': 11},
    ])
    u.pg.insert('b', [
        {'id': 3, 'cc': 'b3', 'parent_k': 11},
        {'id': 4, 'cc': 'b4', 'parent_k': 11},
    ])
    u.pg.insert('c', [
        {'id': 5, 'cc': 'c5', 'parent_k': 11},
        {'id': 6, 'cc': 'c6', 'parent_k': 11},
    ])

    parent_auto = u.cast(gws.Model, root.get('PARENT_AUTO'))
    a = u.cast(gws.Model, root.get('A'))
    b = u.cast(gws.Model, root.get('B'))
    c = u.cast(gws.Model, root.get('C'))

    f = u.feature(parent_auto, k=99, children=[
        u.feature(a, id=1),
        u.feature(a, id=3),
        u.feature(b, id=4),
    ])

    parent_auto.create_feature(f, mc)

    assert u.pg.rows(_SELECT_ALL) == [
        ('a', 1, 99),
        ('a', 2, 11),
        ('a', 3, 99),
        ('b', 3, 11),
        ('b', 4, 99),
        ('c', 5, 11),
        ('c', 6, 11),
    ]


def test_delete(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
    ])
    u.pg.insert('a', [
        {'id': 1, 'cc': 'a1', 'parent_k': 11},
        {'id': 2, 'cc': 'a2', 'parent_k': 11},
        {'id': 3, 'cc': 'a2', 'parent_k': 22},
    ])
    u.pg.insert('b', [
        {'id': 3, 'cc': 'b3', 'parent_k': 11},
        {'id': 4, 'cc': 'b4', 'parent_k': 22},
    ])
    u.pg.insert('c', [
        {'id': 5, 'cc': 'c5', 'parent_k': 22},
        {'id': 6, 'cc': 'c6', 'parent_k': 11},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))

    f = u.feature(parent, id=1)
    parent.delete_feature(f, mc)

    assert u.pg.rows(_SELECT_ALL) == [
        ('a', 1, None),
        ('a', 2, None),
        ('a', 3, 22),
        ('b', 3, None),
        ('b', 4, 22),
        ('c', 5, 22),
        ('c', 6, None),
    ]


def test_create_related(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
    ])
    u.pg.insert('a', [
        {'id': 1, 'cc': 'a1', 'parent_k': 11},
    ])
    u.pg.insert('b', [
        {'id': 3, 'cc': 'b3', 'parent_k': 22},
    ])
    u.pg.insert('c', [
        {'id': 5, 'cc': 'c5', 'parent_k': 33},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    a = u.cast(gws.Model, root.get('A'))
    b = u.cast(gws.Model, root.get('B'))
    c = u.cast(gws.Model, root.get('C'))

    a_f = u.feature(a, id=101)
    a_f.createWithFeatures = [u.feature(parent, id=1)]
    a.create_feature(a_f, mc)

    b_f = u.feature(b, id=201)
    b_f.createWithFeatures = [u.feature(parent, id=1)]
    b.create_feature(b_f, mc)

    c_f = u.feature(c, id=202)
    c_f.createWithFeatures = [u.feature(parent, id=1)]
    c.create_feature(c_f, mc)

    assert u.pg.rows(_SELECT_ALL) == [
        ('a', 1, 11),
        ('a', 101, 11),
        ('b', 3, 22),
        ('b', 201, 11),
        ('c', 5, 33),
        ('c', 202, 11),
    ]
