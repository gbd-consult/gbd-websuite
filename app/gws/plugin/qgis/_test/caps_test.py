import gws
import gws.plugin.qgis.caps as caps


def test_parse_datasource_space_delimited_escaped_quotes():
    text = r"""dbname='db_1' user='user_1' password='a\'b\\c' table="schema_1"."table_1" (geom)"""
    ds = caps.parse_datasource('postgres', text)
    assert ds['dbname'] == 'db_1'
    assert ds['user'] == 'user_1'
    assert ds['password'] == "a'b\\c"
    assert ds['table'] == 'schema_1.table_1'
    assert ds['geometrycolumn'] == 'geom'
    assert ds['provider'] == 'postgres'


def test_parse_datasource_special_table_names():
    cases = [
        (r'"public"."tab (1) ä-ö"', 'public."tab (1) ä-ö"'),
        (r'"public"."a.b"', 'public."a.b"'),
        (r'"my.schema"."t"', '"my.schema".t'),
        (r'"public"."x\"y"', 'public."x""y"'),
        (r'"plain_table"', 'plain_table'),
    ]
    for table, expected in cases:
        ds = caps.parse_datasource('postgres', f'dbname=db_1 table={table} (geom)')
        assert ds['table'] == expected
