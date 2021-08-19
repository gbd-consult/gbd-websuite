import gws.base.db.postgres.driver
import gws.lib.test as test


@test.fixture(scope='module', autouse=True)
def configuration():
    test.setup()

    test.postgres_make_features(
        name='test1',
        geom_type='point',
        columns={
            'a': 'varchar',
            'i': 'int'
        },
        xy=[100, 500],
        crs='EPSG:3857',
        rows=10,
        cols=10,
        gap=100)

    yield

    test.postgres_drop_table('test1')
    test.teardown()


@test.fixture
def conn():
    yield gws.base.db.postgres.driver.Connection(test.postgres_connect_params())


def test_select(conn):
    with conn:
        recs = list(conn.select('SELECT * FROM test1 WHERE id in (1, 2, 3) ORDER BY id'))
    assert [r['i'] for r in recs] == [100, 200, 300]
