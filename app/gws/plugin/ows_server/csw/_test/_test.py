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
            type "csw"
            uid "CSW_1"
            withInspireMeta true
            metadata {DEFAULT_METADATA}
        }

        helpers+ {
            type "xml"
            namespaces+ {
                xmlns "demo"
                uri "http://localhost/_/owsXml/namespace/demo"
            }
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
                type "geojson"
                provider.path "/gws-app/gws/plugin/ows_server/wfs/_test/1.geojson"
                ows.xmlns "demo"
                withSearch true
            }
        }
    """

    yield u.gws_root(cfg, DEFAULT_METADATA=u.DEFAULT_METADATA)


def test_valid_GetCapabilities(root: gws.Root):
    s = u.http.get(
        root,
        '/_/owsService',
        query_string={
            'request': 'GetCapabilities',
            'serviceUid': 'CSW_1',
        },
    )
    gws.u.write_file_b(f'{gws.c.VAR_DIR}/csw_GetCapabilities.xml', s.get_data())
    assert gws.lib.xmlx.validator.validate(s.get_data())


def test_valid_GetRecords(root: gws.Root):
    s = u.http.get(
        root,
        '/_/owsService',
        query_string={
            'request': 'GetRecords',
            'serviceUid': 'CSW_1',
        },
    )
    gws.u.write_file_b(f'{gws.c.VAR_DIR}/csw_GetRecords.xml', s.get_data())
    assert gws.lib.xmlx.validator.validate(s.get_data())
