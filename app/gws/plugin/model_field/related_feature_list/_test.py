import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('parent', {'id': 'int primary key', 'k': 'int', 'pp': 'text'})
    u.pg.create('parent_auto', {'id': 'int primary key generated always as identity', 'k': 'int', 'pp': 'text'})
    u.pg.create('child', {'id': 'int primary key', 'cc': 'text', 'parent_k': 'int'})

    cfg = '''
        models+ { 
            uid "PARENT" type "postgres" tableName "parent"
            fields+ { name "id" type "integer" }
            fields+ { name "k" type "text" }
            fields+ { name "pp" type "text" }
            fields+ { 
                name "children"  
                type relatedFeatureList
                fromColumn "k"
                toModel "CHILD"
                toColumn "parent_k"
            }
        }

        models+ { 
            uid "PARENT_AUTO" type "postgres" tableName "parent_auto"
            fields+ { name "id" type "integer" }
            fields+ { name "k" type "text" }
            fields+ { name "pp" type "text" }
            fields+ { 
                name "children"  
                type relatedFeatureList
                fromColumn "k"
                toModel "CHILD"
                toColumn "parent_k"
            }
        }
        
        models+ { 
            uid "CHILD" type "postgres" tableName "child"
            fields+ { name "id" type "integer" }
            fields+ { name "cc" type "text" }
        }
    '''

    yield u.gws_root(cfg)


def test_find_depth(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
        {'id': 3, 'k': 33, 'pp': 'p3'},
    ])
    u.pg.insert('child', [
        {'id': 4, 'cc': 'a4', 'parent_k': 11},
        {'id': 5, 'cc': 'a5', 'parent_k': 22},
        {'id': 6, 'cc': 'a6', 'parent_k': 11},
        {'id': 7, 'cc': 'a7', 'parent_k': 11},
        {'id': 8, 'cc': 'a8', 'parent_k': 99},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    fs = parent.get_features([1, 2, 3], mc)

    assert [f.get('pp') for f in fs] == ['p1', 'p2', 'p3']

    assert [c.get('cc') for c in fs[0].get('children')] == ['a4', 'a6', 'a7']
    assert [c.get('cc') for c in fs[1].get('children')] == ['a5']
    assert [c.get('cc') for c in fs[2].get('children')] == []


def test_update(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
        {'id': 3, 'k': 33, 'pp': 'p3'},
    ])
    u.pg.insert('child', [
        {'id': 4, 'cc': 'a4', 'parent_k': 11},
        {'id': 5, 'cc': 'a5', 'parent_k': 22},
        {'id': 6, 'cc': 'a6', 'parent_k': 11},
        {'id': 7, 'cc': 'a7', 'parent_k': 33},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    child = u.cast(gws.Model, root.get('CHILD'))

    f = u.feature(parent, id=1, children=[
        u.feature(child, id=4),
        u.feature(child, id=5),
    ])

    parent.update_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [
        (4, 11), (5, 11), (6, None), (7, 33)
    ]


def test_create(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
        {'id': 3, 'k': 33, 'pp': 'p3'},
    ])
    u.pg.insert('child', [
        {'id': 4, 'cc': 'a4', 'parent_k': 11},
        {'id': 5, 'cc': 'a5', 'parent_k': 22},
        {'id': 6, 'cc': 'a6', 'parent_k': 11},
        {'id': 7, 'cc': 'a7', 'parent_k': 33},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    child = u.cast(gws.Model, root.get('CHILD'))

    f = u.feature(parent, id=999, k=99, children=[
        u.feature(child, id=4),
        u.feature(child, id=5),
        u.feature(child, id=6),
    ])

    parent.create_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [
        (4, 99), (5, 99), (6, 99), (7, 33)
    ]


def test_create_auto(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('child', [
        {'id': 4, 'cc': 'a4', 'parent_k': 11},
        {'id': 5, 'cc': 'a5', 'parent_k': 22},
        {'id': 6, 'cc': 'a6', 'parent_k': 11},
        {'id': 7, 'cc': 'a7', 'parent_k': 33},
    ])

    parent_auto = u.cast(gws.Model, root.get('PARENT_AUTO'))
    child = u.cast(gws.Model, root.get('CHILD'))

    f = u.feature(parent_auto, k=99, children=[
        u.feature(child, id=4),
        u.feature(child, id=5),
        u.feature(child, id=6),
    ])

    parent_auto.create_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [
        (4, 99), (5, 99), (6, 99), (7, 33)
    ]


def test_delete(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
    ])
    u.pg.insert('child', [
        {'id': 4, 'cc': 'a4', 'parent_k': 11},
        {'id': 5, 'cc': 'a5', 'parent_k': 22},
        {'id': 6, 'cc': 'a6', 'parent_k': 11},
        {'id': 7, 'cc': 'a7', 'parent_k': 33},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))

    f = u.feature(parent, id=1)
    parent.delete_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [
        (4, None), (5, 22), (6, None), (7, 33),
    ]


def test_create_related(root: gws.Root):
    mc = u.model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
    ])
    u.pg.insert('child', [
        {'id': 4, 'cc': 'a4', 'parent_k': 11},
        {'id': 5, 'cc': 'a5', 'parent_k': 22},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    child = u.cast(gws.Model, root.get('CHILD'))

    child_f = u.feature(child, id=101)
    child_f.createWithFeatures = [u.feature(parent, id=1)]
    child.create_feature(child_f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [
        (4, 11), (5, 22), (101, 11)
    ]
