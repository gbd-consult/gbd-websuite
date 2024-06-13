import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('parent', {'id': 'int primary key', 'k': 'int', 'pp': 'text'})
    u.pg.create('child', {'id': 'int primary key', 'cc': 'text', 'parent_k': 'int'})
    u.pg.create('child_auto', {'id': 'int primary key generated always as identity', 'cc': 'text', 'parent_k': 'int'})

    cfg = '''
        models+ { 
            uid "PARENT" type "postgres" tableName "parent"
            fields+ { name "id" type "integer" }
            fields+ { name "k" type "integer" }
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

        models+ { 
            uid "CHILD_AUTO" type "postgres" tableName "child_auto"
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

    yield u.gws_root(cfg)


def test_find_no_depth(root: gws.Root):
    mc = u.gws_model_context(maxDepth=0)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
    ])
    u.pg.insert('child', [
        {'id': 1, 'cc': 'c1', 'parent_k': 1},
        {'id': 2, 'cc': 'c2', 'parent_k': 1},
        {'id': 3, 'cc': 'c3', 'parent_k': 1},
    ])

    child = u.cast(gws.Model, root.get('CHILD'))
    fs = child.get_features([2, 3], mc)

    assert set(f.get('cc') for f in fs) == {'c2', 'c3'}
    assert fs[0].get('parent') is None
    assert fs[1].get('parent') is None


def test_find_depth(root: gws.Root):
    mc = u.gws_model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
    ])
    u.pg.insert('child', [
        {'id': 1, 'cc': 'c1', 'parent_k': 11},
        {'id': 2, 'cc': 'c2', 'parent_k': 11},
        {'id': 3, 'cc': 'c3', 'parent_k': 22},
        {'id': 4, 'cc': 'c4', 'parent_k': 99},
    ])

    child = u.cast(gws.Model, root.get('CHILD'))
    fs = child.get_features([1, 2, 3, 4], mc)

    assert set(f.get('cc') for f in fs) == {'c1', 'c2', 'c3', 'c4'}
    assert fs[0].get('parent').get('pp') == 'p1'
    assert fs[1].get('parent').get('pp') == 'p1'
    assert fs[2].get('parent').get('pp') == 'p2'
    assert fs[3].get('parent') is None


def test_update(root: gws.Root):
    mc = u.gws_model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
        {'id': 3, 'k': 33, 'pp': 'p3'},
    ])
    u.pg.insert('child', [
        {'id': 1, 'cc': 'c1', 'parent_k': 11},
        {'id': 2, 'cc': 'c2', 'parent_k': 11},
        {'id': 3, 'cc': 'c3', 'parent_k': 11},
        {'id': 4, 'cc': 'c4', 'parent_k': 11},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    child = u.cast(gws.Model, root.get('CHILD'))

    f = u.feature(child, id=2, parent=u.feature(parent, id=3))
    child.update_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [(1, 11), (2, 33), (3, 11), (4, 11)]

    f = u.feature(child, id=3, parent=u.feature(parent, id=99))
    child.update_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [(1, 11), (2, 33), (3, None), (4, 11)]

    f = u.feature(child, id=4)
    child.update_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [(1, 11), (2, 33), (3, None), (4, 11)]


def test_create(root: gws.Root):
    mc = u.gws_model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
        {'id': 3, 'k': 33, 'pp': 'p3'},
    ])
    u.pg.insert('child', [
        {'id': 1, 'cc': 'c1', 'parent_k': 11},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    child = u.cast(gws.Model, root.get('CHILD'))

    f = u.feature(child, id=101, parent=u.feature(parent, id=1))
    child.create_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [(1, 11), (101, 11)]

    f = u.feature(child, id=102)
    child.create_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [(1, 11), (101, 11), (102, None)]


def test_create_auto(root: gws.Root):
    mc = u.gws_model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
        {'id': 2, 'k': 22, 'pp': 'p2'},
        {'id': 3, 'k': 33, 'pp': 'p3'},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    child_auto = u.cast(gws.Model, root.get('CHILD_AUTO'))

    f = u.feature(child_auto, parent=u.feature(parent, id=1))
    child_auto.create_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child_auto ORDER BY id')
    assert rows == [(1, 11)]

    f = u.feature(child_auto)
    child_auto.create_feature(f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child_auto ORDER BY id')
    assert rows == [(1, 11), (2, None)]


def test_create_related(root: gws.Root):
    mc = u.gws_model_context(maxDepth=1)

    u.pg.insert('parent', [
        {'id': 1, 'k': 11, 'pp': 'p1'},
    ])
    u.pg.insert('child', [
        {'id': 1, 'cc': 'c1', 'parent_k': 11},
        {'id': 2, 'cc': 'c2', 'parent_k': 22},
    ])

    parent = u.cast(gws.Model, root.get('PARENT'))
    child = u.cast(gws.Model, root.get('CHILD'))

    parent_f = u.feature(parent, id=9, k=99)
    parent_f.createWithFeatures = [u.feature(child, id=2)]
    parent.create_feature(parent_f, mc)

    rows = u.pg.rows('SELECT id, parent_k FROM child ORDER BY id')
    assert rows == [(1, 11), (2, 99)]
