"""Tests the cql builder module by running its output against postgres."""

import pytest

import gws
import gws.lib.cql as cql
import gws.lib.sa as sa
import gws.test.util as u

TABLE_NAME = 'cql_table'
TABLE_NAME_3857 = 'cql_table_3857'


def _table(name, srid):
    return sa.Table(
        name,
        sa.MetaData(),
        sa.Column('id', sa.Integer),
        sa.Column('a_int', sa.Integer),
        sa.Column('a_float', sa.Float),
        sa.Column('a_text', sa.Text),
        sa.Column('a_bool', sa.Boolean),
        sa.Column('a_date', sa.Date),
        sa.Column('a_time', sa.TIMESTAMP(timezone=True)),
        sa.Column('a_time_end', sa.TIMESTAMP(timezone=True)),
        sa.Column('a_arr', sa.ARRAY(sa.Text)),
        sa.Column('a_geom', sa.geo.Geometry(srid=srid)),
    )


@u.fixture(scope='module')
def table():
    cols = {
        'id': 'int primary key',
        'a_int': 'int',
        'a_float': 'float',
        'a_text': 'text',
        'a_bool': 'bool',
        'a_date': 'date',
        'a_time': 'timestamptz',
        'a_time_end': 'timestamptz',
        'a_arr': 'text[]',
        'a_geom': 'geometry(Geometry, 4326)',
    }
    u.pg.exec('CREATE EXTENSION IF NOT EXISTS unaccent')
    u.pg.create(TABLE_NAME, cols)
    u.pg.exec(f"""
        INSERT INTO {TABLE_NAME} VALUES
            (1, 10, 1.5, 'alpha', true,  '2020-01-01', '2020-01-01T10:00:00Z', '2020-01-01T12:00:00Z', '{{red,green}}',  ST_GeomFromText('POINT(1 1)', 4326)),
            (2, 20, 2.5, 'beta',  false, '2021-06-15', '2021-06-15T12:30:00Z', '2021-06-15T18:00:00Z', '{{green,blue}}', ST_GeomFromText('POINT(5 5)', 4326)),
            (3, 30, 3.5, 'gamma', true,  '2022-12-31', '2022-12-31T23:59:59Z', '2023-01-01T06:00:00Z', '{{blue}}',       ST_GeomFromText('POINT(9 9)', 4326)),
            (4, null, null, null, null,  null,         null,                   null,                   null,            null)
    """)
    yield _table(TABLE_NAME, 4326)


@u.fixture(scope='module')
def table_3857():
    cols = {
        'id': 'int primary key',
        'a_int': 'int',
        'a_float': 'float',
        'a_text': 'text',
        'a_bool': 'bool',
        'a_date': 'date',
        'a_time': 'timestamptz',
        'a_time_end': 'timestamptz',
        'a_arr': 'text[]',
        'a_geom': 'geometry(Geometry, 3857)',
    }
    u.pg.create(TABLE_NAME_3857, cols)
    u.pg.exec(f"""
        INSERT INTO {TABLE_NAME_3857} (id, a_geom) VALUES
            (1, ST_Transform(ST_GeomFromText('POINT(1 1)', 4326), 3857)),
            (2, ST_Transform(ST_GeomFromText('POINT(5 5)', 4326), 3857)),
            (3, ST_Transform(ST_GeomFromText('POINT(9 9)', 4326), 3857))
    """)
    yield _table(TABLE_NAME_3857, 3857)


class _BuilderTo3857(cql.SqlBuilder):
    """Mimics a model which stores geometries in a projected crs."""

    def build_wkt(self, args):
        return sa.func.ST_Transform(super().build_wkt(args), 3857)

    def build_bbox(self, args):
        return sa.func.ST_Transform(super().build_bbox(args), 3857)


def _ids(tbl, src, cls=cql.SqlBuilder):
    cond = cls(tbl).build(cql.parse(src))
    sel = sa.select(tbl.c.id).where(cond).order_by(tbl.c.id)
    with u.pg.connect() as conn:
        return [r[0] for r in conn.execute(sel)]


##


@pytest.mark.parametrize(
    'src,expected',
    [
        ('a_int = 20', [2]),
        ('a_int <> 20', [1, 3]),
        ('a_int != 20', [1, 3]),
        ('a_int > 15', [2, 3]),
        ('a_int >= 20', [2, 3]),
        ('a_int < 30', [1, 2]),
        ('a_int <= 10', [1]),
        ('a_float = 2.5', [2]),
        ('a_float > 2', [2, 3]),
        ("a_text = 'beta'", [2]),
        ('a_bool = true', [1, 3]),
        ('a_bool = false', [2]),
    ],
)
def test_comparison(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ('a_int + 5 = 25', [2]),
        ('a_int - 5 = 25', [3]),
        ('a_int * 2 = 60', [3]),
        ('a_int / 10 = 2', [2]),
        ('a_int % 20 = 0', [2]),
        ('a_int ^ 2 = 400', [2]),
        ('a_int + a_float = 11.5', [1]),
        ('a_int * 2 > 40', [3]),
    ],
)
def test_arithmetic(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ('a_int = 10 AND a_bool = true', [1]),
        ('a_int = 10 OR a_int = 30', [1, 3]),
        ('NOT (a_int = 10)', [2, 3]),
        ('NOT NOT (a_int = 10)', [1]),
        ("a_int > 5 AND a_text LIKE 'b%' OR a_int = 30", [2, 3]),
        ('(a_int = 10 OR a_int = 20) AND a_bool = true', [1]),
    ],
)
def test_boolean(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ("a_text LIKE 'b%'", [2]),
        ("a_text LIKE '%ma'", [3]),
        ("a_text LIKE '_eta'", [2]),
        ("a_text NOT LIKE 'b%'", [1, 3]),
        ("a_text LIKE 'B%'", []),
    ],
)
def test_like(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ('a_int BETWEEN 15 AND 35', [2, 3]),
        ('a_int BETWEEN 10 AND 10', [1]),
        ('a_int NOT BETWEEN 15 AND 35', [1]),
        ('a_date BETWEEN 2021-01-01 AND 2021-12-31', [2]),
    ],
)
def test_between(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ('a_int IN (10, 30)', [1, 3]),
        ('a_int IN [10, 30]', [1, 3]),
        ('a_int NOT IN (10, 30)', [2]),
        ("a_text IN ('beta', 'gamma')", [2, 3]),
        ('a_int IN (99)', []),
    ],
)
def test_in(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ('a_int IS NULL', [4]),
        ('a_int IS NOT NULL', [1, 2, 3]),
        ('a_text IS NULL', [4]),
        ('a_geom IS NULL', [4]),
    ],
)
def test_null(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ('a_date = 2021-06-15', [2]),
        ('a_date > 2021-01-01', [2, 3]),
        ("a_date = DATE('2021-06-15')", [2]),
        ("a_date < DATE('2021-01-01')", [1]),
    ],
)
def test_date(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ('a_time = 2021-06-15T12:30:00Z', [2]),
        ('a_time > 2021-01-01T00:00:00Z', [2, 3]),
        ("a_time = TIMESTAMP('2021-06-15T12:30:00Z')", [2]),
        ("a_time < TIMESTAMP('2021-01-01T00:00:00Z')", [1]),
    ],
)
def test_timestamp(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ('S_INTERSECTS(a_geom, POINT(5 5))', [2]),
        ('S_INTERSECTS(a_geom, POINT(7 7))', []),
        ('S_INTERSECTS(a_geom, POLYGON((0 0, 6 0, 6 6, 0 6, 0 0)))', [1, 2]),
        ('S_INTERSECTS(a_geom, BBOX(0, 0, 6, 6))', [1, 2]),
        ('S_INTERSECTS(a_geom, BBOX(0, 0, 100, 100))', [1, 2, 3]),
        ('S_INTERSECTS(a_geom, BBOX(50, 50, 60, 60))', []),
        ('S_INTERSECTS(a_geom, POINT(5 5)) AND a_int = 20', [2]),
        ('NOT S_INTERSECTS(a_geom, BBOX(0, 0, 6, 6))', [3]),
    ],
)
def test_spatial(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ('S_INTERSECTS(a_geom, POINT(5 5))', [2]),
        ('S_INTERSECTS(a_geom, BBOX(0, 0, 6, 6))', [1, 2]),
        ('S_INTERSECTS(a_geom, BBOX(50, 50, 60, 60))', []),
    ],
)
def test_spatial_reprojected(table_3857, src, expected):
    assert _ids(table_3857, src, _BuilderTo3857) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ('S_EQUALS(a_geom, POINT(5 5))', [2]),
        ('S_CONTAINS(a_geom, POINT(5 5))', [2]),
        ('S_WITHIN(a_geom, POLYGON((0 0, 6 0, 6 6, 0 6, 0 0)))', [1, 2]),
        ('S_DISJOINT(a_geom, BBOX(0, 0, 6, 6))', [3]),
        ('S_TOUCHES(a_geom, POLYGON((1 1, 5 1, 5 5, 1 5, 1 1)))', [1, 2]),
        ('S_CROSSES(a_geom, POLYGON((0 0, 6 0, 6 6, 0 6, 0 0)))', []),
        ('S_OVERLAPS(a_geom, POLYGON((0 0, 6 0, 6 6, 0 6, 0 0)))', []),
    ],
)
def test_spatial_predicates(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ("CASEI(a_text) = 'alpha'", [1]),
        ("CASEI(a_text) = 'ALPHA'", []),
        ("CASEI(a_text) = CASEI('ALPHA')", [1]),
        ("CASEI(a_text) LIKE CASEI('AL%')", [1]),
        ("a_text LIKE CASEI('BETA')", [2]),
        ('ACCENTI(a_text) = a_text', [1, 2, 3]),
        ("ACCENTI('grün') = 'grun'", [1, 2, 3, 4]),
        ("CASEI(ACCENTI('GRÜN')) = 'grun'", [1, 2, 3, 4]),
    ],
)
def test_casei_accenti(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ("A_CONTAINS(a_arr, ('red'))", [1]),
        ("A_CONTAINS(a_arr, ('green'))", [1, 2]),
        ("A_CONTAINS(a_arr, ('green', 'blue'))", [2]),
        ("A_CONTAINS(a_arr, ['red'])", [1]),
        ('A_CONTAINS(a_arr, ())', [1, 2, 3]),
        ("A_CONTAINEDBY(a_arr, ('red', 'green', 'blue'))", [1, 2, 3]),
        ("A_CONTAINEDBY(a_arr, ('blue'))", [3]),
        ("A_EQUALS(a_arr, ('green', 'red'))", [1]),
        ("A_EQUALS(a_arr, ('blue'))", [3]),
        ("A_OVERLAPS(a_arr, ('blue'))", [2, 3]),
        ("A_OVERLAPS(a_arr, ('yellow'))", []),
    ],
)
def test_array(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ("T_EQUALS(a_time, TIMESTAMP('2021-06-15T12:30:00Z'))", [2]),
        ("T_AFTER(a_time, TIMESTAMP('2021-01-01T00:00:00Z'))", [2, 3]),
        ("T_BEFORE(a_time, TIMESTAMP('2021-01-01T00:00:00Z'))", [1]),
        ("T_AFTER(a_date, DATE('2021-01-01'))", [2, 3]),
        ("T_BEFORE(a_date, DATE('2021-01-01'))", [1]),
    ],
)
def test_temporal_instants(table, src, expected):
    assert _ids(table, src) == expected


@pytest.mark.parametrize(
    'src,expected',
    [
        ("T_DURING(a_time, INTERVAL('2021-01-01T00:00:00Z', '2022-01-01T00:00:00Z'))", [2]),
        ("T_DISJOINT(a_time, INTERVAL('2021-01-01T00:00:00Z', '2022-01-01T00:00:00Z'))", [1, 3]),
        ("T_INTERSECTS(a_time, INTERVAL('2021-01-01T00:00:00Z', '2022-01-01T00:00:00Z'))", [2]),
        ("T_AFTER(a_time, INTERVAL('..', '2021-01-01T00:00:00Z'))", [2, 3]),
        ("T_BEFORE(a_time, INTERVAL('2021-01-01T00:00:00Z', '..'))", [1]),
        ("T_EQUALS(INTERVAL(a_time, a_time_end), INTERVAL('2020-01-01T10:00:00Z', '2020-01-01T12:00:00Z'))", [1]),
        ("T_CONTAINS(INTERVAL(a_time, a_time_end), TIMESTAMP('2020-01-01T11:00:00Z'))", [1]),
        ("T_MEETS(INTERVAL(a_time, a_time_end), INTERVAL('2020-01-01T12:00:00Z', '2020-01-01T15:00:00Z'))", [1]),
        ("T_METBY(INTERVAL(a_time, a_time_end), INTERVAL('2020-01-01T08:00:00Z', '2020-01-01T10:00:00Z'))", [1]),
        ("T_OVERLAPS(INTERVAL(a_time, a_time_end), INTERVAL('2020-01-01T11:00:00Z', '2020-01-01T15:00:00Z'))", [1]),
        ("T_OVERLAPPEDBY(INTERVAL(a_time, a_time_end), INTERVAL('2020-01-01T09:00:00Z', '2020-01-01T11:00:00Z'))", [1]),
        ("T_STARTS(INTERVAL(a_time, a_time_end), INTERVAL('2020-01-01T10:00:00Z', '2020-01-01T18:00:00Z'))", [1]),
        ("T_STARTEDBY(INTERVAL(a_time, a_time_end), INTERVAL('2020-01-01T10:00:00Z', '2020-01-01T11:00:00Z'))", [1]),
        ("T_FINISHES(INTERVAL(a_time, a_time_end), INTERVAL('2020-01-01T08:00:00Z', '2020-01-01T12:00:00Z'))", [1]),
        ("T_FINISHEDBY(INTERVAL(a_time, a_time_end), INTERVAL('2020-01-01T11:00:00Z', '2020-01-01T12:00:00Z'))", [1]),
    ],
)
def test_temporal_intervals(table, src, expected):
    assert _ids(table, src) == expected


def test_interval_outside_temporal_predicate(table):
    with pytest.raises(cql.BuildError):
        _ids(table, "a_time = INTERVAL('2020-01-01T00:00:00Z', '2020-01-02T00:00:00Z')")


##


def test_unknown_column(table):
    with pytest.raises(cql.BuildError):
        _ids(table, 'nosuch = 1')


def test_unknown_function(table):
    with pytest.raises(cql.BuildError):
        _ids(table, 'NOSUCH(a_geom, POINT(1 1))')


def test_unknown_operator(table):
    with pytest.raises(cql.BuildError):
        cql.SqlBuilder(table).build(['NOSUCH', [cql.Node.INT, 1]])


def test_non_literal_argument(table):
    with pytest.raises(cql.BuildError):
        _ids(table, 'S_INTERSECTS(a_geom, BBOX(a_int, 2, 3, 4))')
