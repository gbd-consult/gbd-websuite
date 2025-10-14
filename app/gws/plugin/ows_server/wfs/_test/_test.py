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
            type "wfs"
            uid "WFS_1"
        }
        owsServices+ {
            type "wfs"
            uid "WFS_1_INSPIRE"
            withInspireMeta true
            metadata {DEFAULT_METADATA}
        }
        helpers+ {
            type "xml"
            namespaces+ {
                xmlns "demo"
                uri "https://gws-dev.gbd-consult.de/_/owsXml/namespace/demo"
                schemaLocation "https://gws-dev.gbd-consult.de/_/owsXml/namespace/demo"
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
            'serviceUid': 'WFS_1',
            'projectUid': 'PROJECT_1',
            'version': '2.0.2',
        },
    )
    gws.u.write_file_b('/tmp/wfs_GetCapabilities.xml', s.get_data())
    assert gws.lib.xmlx.validator.validate(s.get_data())


def test_valid_GetCapabilities_inspire(root: gws.Root):
    s = u.http.get(
        root,
        '/_/owsService',
        query_string={
            'request': 'GetCapabilities',
            'serviceUid': 'WFS_1_INSPIRE',
            'projectUid': 'PROJECT_1',
            'version': '2.0.2',
        },
    )
    gws.u.write_file_b('/tmp/wfs_GetCapabilities_inspire.xml', s.get_data())
    assert gws.lib.xmlx.validator.validate(s.get_data())


def test_valid_GetFeature(root: gws.Root):
    s = u.http.get(
        root,
        '/_/owsService',
        query_string={
            'request': 'GetFeature',
            'serviceUid': 'WFS_1',
            'projectUid': 'PROJECT_1',
            'version': '2.0.2',
        },
    )
    gws.u.write_file_b('/tmp/wfs_GetFeature.xml', s.get_data())
    assert gws.lib.xmlx.validator.validate(s.get_data())

def test_valid_DescribeFeatureType(root: gws.Root):
    s = u.http.get(
        root,
        '/_/owsService',
        query_string={
            'request': 'DescribeFeatureType',
            'serviceUid': 'WFS_1',
            'projectUid': 'PROJECT_1',
            'version': '2.0.2',
        },
    )
    gws.u.write_file_b('/tmp/wfs_DescribeFeatureType.xml', s.get_data())
    assert gws.lib.xmlx.validator.validate(s.get_data())
