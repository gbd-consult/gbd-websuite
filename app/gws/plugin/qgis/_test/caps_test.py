import gws
import gws.lib.xmlx
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


def _layer_el(xml):
    return gws.lib.xmlx.from_string(f'<maplayer>{xml}</maplayer>')


def test_layer_opacity_vector():
    el = _layer_el('<layerOpacity>0.3</layerOpacity>')
    assert caps._map_layer_opacity(el) == 0.3


def test_layer_opacity_raster():
    el = _layer_el('<pipe><rasterrenderer type="singlebandcolordata" opacity="0.5"/></pipe>')
    assert caps._map_layer_opacity(el) == 0.5


def test_layer_opacity_default():
    el = _layer_el('<layername>layer_1</layername>')
    assert caps._map_layer_opacity(el) == 1
