import gws
import gws.lib.test.util as u


@u.fixture(scope='module')
def gws_root():
    u.pg_create('a', {'id': 'int primary key', 'ka': 'int', 'sa': 'text'})
    u.pg_create('b', {'id': 'int primary key', 'kb': 'int', 'sb': 'text'})
    u.pg_create('x', {'xa': 'int', 'xb': 'int'})

    cfg = '''
        models+ { 
            uid "A" type "postgres" tableName "a"
            fields+ { name "id" type "integer" }
            fields+ { name "ka" type "integer" }
            fields+ { name "sa" type "text" }
            fields+ { 
                name "linked"  
                type relatedLinkedFeatureList 
                fromColumn "ka"
                toModel "B" 
                toColumn "kb"
                linkTableName "x"
                linkFromColumn "xa"
                linkToColumn "xb"
            }
        }
        
        models+ { 
            uid "B" type "postgres" tableName "b"
            fields+ { name "id" type "integer" }
            fields+ { name "kb" type "integer" }
            fields+ { name "sb" type "text" }
        }
        
    '''

    yield u.gws_configure(cfg)


def test_find_depth(gws_root):
    mc = u.gws_model_context(maxDepth=1)

    u.pg_insert('a', [
        {'id': 1, 'ka': 11, 'sa': 'a1'},
        {'id': 2, 'ka': 22, 'sa': 'a2'},
        {'id': 3, 'ka': 33, 'sa': 'a3'},
    ])
    u.pg_insert('b', [
        {'id': 4, 'kb': 400, 'sb': 'b4'},
        {'id': 5, 'kb': 500, 'sb': 'b5'},
        {'id': 6, 'kb': 600, 'sb': 'b6'},
    ])
    u.pg_insert('x', [
        {'xa': 11, 'xb': 400},
        {'xa': 11, 'xb': 500},
        {'xa': 11, 'xb': 600},
        {'xa': 22, 'xb': 400},
        {'xa': 22, 'xb': 500},
    ])

    ma = u.model(gws_root, 'A')
    fs = ma.get_features([1, 2, 3], mc)

    assert [f.get('sa') for f in fs] == ['a1', 'a2', 'a3']

    assert [c.get('sb') for c in fs[0].get('linked')] == ['b4', 'b5', 'b6']
    assert [c.get('sb') for c in fs[1].get('linked')] == ['b4', 'b5']
    assert [c.get('sb') for c in fs[2].get('linked')] == []


def test_update(gws_root):
    mc = u.gws_model_context(maxDepth=1)

    u.pg_insert('a', [
        {'id': 1, 'ka': 11, 'sa': 'a1'},
        {'id': 2, 'ka': 22, 'sa': 'a2'},
        {'id': 3, 'ka': 33, 'sa': 'a3'},
    ])
    u.pg_insert('b', [
        {'id': 4, 'kb': 400, 'sb': 'b4'},
        {'id': 5, 'kb': 500, 'sb': 'b5'},
        {'id': 6, 'kb': 600, 'sb': 'b6'},
    ])
    u.pg_insert('x', [
        {'xa': 11, 'xb': 400},
        {'xa': 11, 'xb': 500},
        {'xa': 22, 'xb': 400},
        {'xa': 22, 'xb': 500},
    ])

    ma = u.model(gws_root, 'A')
    mb = u.model(gws_root, 'B')

    f = u.feature(ma, id=1, linked=[
        u.feature(mb, id=4),
        u.feature(mb, id=6),
    ])

    ma.update_feature(f, mc)

    assert u.pg_rows('SELECT xa,xb FROM x ORDER BY xa,xb') == [
        (11, 400),
        (11, 600),
        (22, 400),
        (22, 500),
    ]


def test_create(gws_root):
    mc = u.gws_model_context(maxDepth=1)

    u.pg_insert('a', [
        {'id': 1, 'ka': 11, 'sa': 'a1'},
        {'id': 2, 'ka': 22, 'sa': 'a2'},
        {'id': 3, 'ka': 33, 'sa': 'a3'},
    ])
    u.pg_insert('b', [
        {'id': 4, 'kb': 400, 'sb': 'b4'},
        {'id': 5, 'kb': 500, 'sb': 'b5'},
        {'id': 6, 'kb': 600, 'sb': 'b6'},
    ])
    u.pg_insert('x', [
        {'xa': 11, 'xb': 400},
        {'xa': 22, 'xb': 500},
    ])

    ma = u.model(gws_root, 'A')
    mb = u.model(gws_root, 'B')

    f = u.feature(ma, id=9, ka=99, linked=[
        u.feature(mb, id=4),
        u.feature(mb, id=5),
    ])

    ma.create_feature(f, mc)

    assert u.pg_rows('SELECT xa,xb FROM x ORDER BY xa,xb') == [
        (11, 400),
        (22, 500),
        (99, 400),
        (99, 500),
    ]


def test_create_related(gws_root):
    mc = u.gws_model_context(maxDepth=1)

    u.pg_insert('a', [
        {'id': 1, 'ka': 11, 'sa': 'a1'},
        {'id': 2, 'ka': 22, 'sa': 'a2'},
        {'id': 3, 'ka': 33, 'sa': 'a3'},
    ])
    u.pg_insert('b', [
        {'id': 4, 'kb': 400, 'sb': 'b4'},
        {'id': 5, 'kb': 500, 'sb': 'b5'},
    ])
    u.pg_insert('x', [
        {'xa': 11, 'xb': 400},
        {'xa': 11, 'xb': 500},
    ])

    ma = u.model(gws_root, 'A')
    mb = u.model(gws_root, 'B')

    b_f = u.feature(mb, id=9, kb=999)
    b_f.createWithFeatures = [
        u.feature(ma, id=1),
        u.feature(ma, id=2),
    ]

    mb.create_feature(b_f, mc)

    assert u.pg_rows('SELECT xa,xb FROM x ORDER BY xa,xb') == [
        (11, 400),
        (11, 500),
        (11, 999),
        (22, 999),
    ]
