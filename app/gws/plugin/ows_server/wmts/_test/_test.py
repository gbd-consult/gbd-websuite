import gws
import gws.test.util as u
import gws.lib.xmlx.validator


@u.fixture(scope='module')
def root():
    cfg = """
        permissions.all "allow all"

        actions [
            { type "ows" }
        ]
        
        owsServices+ {
            type "wmts"
            uid "WMTS_1"
        }
        owsServices+ {
            type "wmts"
            uid "WMTS_1_INSPIRE"
            withInspireMeta true
            metadata {DEFAULT_METADATA}
        }
        projects+ {
            uid "PROJECT_1"
            metadata.abstract "ABSTRACT"
            metadata.keywords ["aaa" "bbb"]
            map.extent [0 0 100 200]
            map.srs "EPSG:3857"
            map.layers+ {
                uid "LAYER_1"
                title "LAYER_1"
                type "raster"
                provider.paths ["/gws-app/gws/plugin/ows_server/wms/_test/300x400.png"]
            }
        }
    """
    yield u.gws_root(cfg, DEFAULT_METADATA=u.DEFAULT_METADATA)

def test_valid_GetCapabilities(root: gws.Root):
    s = u.http.get(root, '/_/owsService', query_string={
        'request': "GetCapabilities",
        'serviceUid': "WMTS_1",
        'projectUid': "PROJECT_1",
    })
    gws.u.write_file_b('/tmp/wmts_GetCapabilities.xml', s.get_data())
    assert gws.lib.xmlx.validator.validate(s.get_data())

def xxxxtest_valid_GetCapabilities_inspire(root: gws.Root):
    s = u.http.get(root, '/_/owsService', query_string={
        'request': "GetCapabilities",
        'serviceUid': "WMTS_1_INSPIRE",
        'projectUid': "PROJECT_1",
    })
    gws.u.write_file_b('/tmp/wmts_GetCapabilities_inspire.xml', s.get_data())
    assert gws.lib.xmlx.validator.validate(s.get_data())
