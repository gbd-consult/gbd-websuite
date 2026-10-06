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
