import gws
import gws.test.util as u
import gws.lib.xmlx
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

    yield u.gws_root(cfg, DEFAULT_METADATA=u.metadata.DEFAULT)


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


@u.fixture(scope='module')
def root_with_record():
    cfg = """
        permissions.all "allow all"

        actions [
            { type "ows" }
        ]

        owsServices+ {
            type "csw"
            uid "CSW_1"
            metadata {DEFAULT_METADATA}
        }

        projects+ {
            uid "PROJECT_1"
            metadata.catalogUid "project_1"
            map.extent [0 0 100 200]
            map.srs "EPSG:3857"
            map.layers+ {
                uid "LAYER_1"
                title "LAYER_1"
                type "geojson"
                provider.path "/gws-app/gws/plugin/ows_server/wfs/_test/1.geojson"
            }
        }
    """

    yield u.gws_root(cfg, DEFAULT_METADATA=u.metadata.DEFAULT)


def test_GetRecordById(root_with_record: gws.Root):
    s = u.http.get(
        root_with_record,
        '/_/owsService',
        query_string={
            'request': 'GetRecordById',
            'serviceUid': 'CSW_1',
            'id': 'project_1',
        },
    )
    xml = s.get_data()
    el = gws.lib.xmlx.from_string(xml, gws.XmlOptions(removeNamespaces=True))
    assert el.name == 'GetRecordByIdResponse'
    md_list = el.findall('MD_Metadata')
    assert len(md_list) == 1
    assert md_list[0].find('language') is not None


def test_GetRecordById_unknown_id(root_with_record: gws.Root):
    s = u.http.get(
        root_with_record,
        '/_/owsService',
        query_string={
            'request': 'GetRecordById',
            'serviceUid': 'CSW_1',
            'id': 'unknown_1',
        },
    )
    xml = s.get_data()
    el = gws.lib.xmlx.from_string(xml, gws.XmlOptions(removeNamespaces=True))
    assert el.name == 'GetRecordByIdResponse'
    assert el.findall('MD_Metadata') == []
