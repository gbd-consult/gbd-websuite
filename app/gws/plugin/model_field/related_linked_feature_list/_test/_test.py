import gws
import gws.lib.test.util as u


@u.fixture(scope='module')
def gws_root():
    u.pg_create_table(
        'a',
        {'id': 'int primary key', 'ka': 'int', 'sa': 'text'},
        {'id': 1, 'ka': 11, 'sa': 'a1'},
        {'id': 2, 'ka': 22, 'sa': 'a2'},
        {'id': 3, 'ka': 33, 'sa': 'a3'},
        {'id': 4, 'ka': 44, 'sa': 'a4'},
        {'id': 5, 'ka': 55, 'sa': 'a5'},
    )
    u.pg_create_table(
        'b',
        {'id': 'int primary key', 'kb': 'int', 'sb': 'text'},
        {'id': 1, 'kb': 111, 'sb': 'b1'},
        {'id': 2, 'kb': 222, 'sb': 'b2'},
        {'id': 3, 'kb': 333, 'sb': 'b3'},
        {'id': 4, 'kb': 444, 'sb': 'b4'},
        {'id': 5, 'kb': 555, 'sb': 'b5'},
    )
    u.pg_create_table(
        'link',
        {'link_a': 'int', 'link_b': 'int'},
    )

    cfg = '''
        database.providers+ { type "postgres" serviceName "gws_test_postgres" } 
        
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
                linkTableName "link"
                linkFromColumn "link_a"
                linkToColumn "link_b"
            }
        }
        
        models+ { 
            uid "B" type "postgres" tableName "b"
            fields+ { name "id" type "integer" }
            fields+ { name "sb" type "text" }
            # fields+ { 
            #     name "b_to_a"  
            #     type relatedLinkedFeatureList 
            #     relationship {
            #         fromColumn "kb"
            #         toModel "A" toColumn "ka"
            #         linkModel "LINK"
            #         linkFromColumn "link_kb"
            #         linkToColumn "link_ka"
            #     }
            # }
        }
        
    '''

    yield u.gws_configure(cfg)


def test_find_depth(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        DELETE FROM link;
        INSERT INTO link VALUES
            (11, 111),
            (11, 333),
            (11, 444),
            (22, 111),
            (22, 444),
            (22, 555),
            (44, 111),
            (44, 555)
    ''')

    ma = u.model(gws_root, 'A')
    fs = ma.find_features(gws.SearchQuery(uids=[1, 2, 3]), mc)

    assert set(f.get('sa') for f in fs) == {'a1', 'a2', 'a3'}

    assert set(c.get('sb') for c in fs[0].get('linked')) == {'b1', 'b3', 'b4'}
    assert set(c.get('sb') for c in fs[1].get('linked')) == {'b1', 'b4', 'b5'}
    assert set(c.get('sb') for c in fs[2].get('linked')) == set()


def test_update(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        DELETE FROM link;
        INSERT INTO link VALUES
            (11, 111),
            (11, 333),
            (11, 444),
            (22, 111),
            (22, 444),
            (22, 555),
            (33, 111),
            (33, 555),
            (44, 111),
            (44, 222)
    ''')

    ma = u.model(gws_root, 'A')
    mb = u.model(gws_root, 'B')

    fs_in = [
        u.feature(ma, id=1),
        u.feature(ma, id=2),
        u.feature(ma, id=3),
    ]
    fs_in[0].set('linked', [
        u.feature(mb, id=1),
        u.feature(mb, id=2),
        u.feature(mb, id=3),
    ])
    fs_in[1].set('linked', [
        u.feature(mb, id=4),
        u.feature(mb, id=5),
    ])
    fs_in[2].set('linked', [

    ])

    ma.update_features(fs_in, mc)

    assert u.pg_rows('SELECT link_a,link_b FROM link ORDER BY link_a,link_b') == [
        (11, 111),
        (11, 222),
        (11, 333),
        (22, 444),
        (22, 555),
        (44, 111),
        (44, 222),
    ]


def test_create(gws_root):
    mc = gws.ModelContext(
        user=u.gws_system_user(),
        mode=gws.ModelMode.edit,
        maxDepth=1,
    )
    u.pg_exec('''
        DELETE FROM link;
        INSERT INTO link VALUES
            (11, 111),
            (22, 222)
    ''')

    ma = u.model(gws_root, 'A')
    mb = u.model(gws_root, 'B')

    fs_in = [
        u.feature(ma, id=101, ka=1111),
        u.feature(ma, id=102, ka=2222),
    ]
    fs_in[0].set('linked', [
        u.feature(mb, id=1),
        u.feature(mb, id=2),
        u.feature(mb, id=3),
    ])
    fs_in[1].set('linked', [
        u.feature(mb, id=4),
        u.feature(mb, id=5),
    ])

    ma.create_features(fs_in, mc)

    assert u.pg_rows('SELECT link_a,link_b FROM link ORDER BY link_a,link_b') == [
        (11, 111),
        (22, 222),
        (1111, 111),
        (1111, 222),
        (1111, 333),
        (2222, 444),
        (2222, 555),
    ]
