"""Tests for the qfieldcloud capabilities parser."""

from typing import cast

import gws
import gws.test.util as u

from gws.plugin.qfieldcloud import action as action_mod, caps
from gws.plugin.qfieldcloud._test import util as tu

CONFIG = """
    projects+ {
        uid "PROJECT_1"
        access "allow all"
        actions+ {
            type "qfieldcloud"
            uid "ACTION_1"
            access "allow all"
            projects+ {
                uid "QFC_1"
                title "QField Test"
                access "allow all"
                provider.path {QGS_PATH}
                {MODELS}
            }
        }
    }
"""


def _root(qgs_path, models=''):
    return u.gws_root(CONFIG, QGS_PATH=repr(qgs_path), MODELS=models)


def _caps(root, uid='QFC_1') -> caps.Caps:
    act = cast(action_mod.Object, root.get('ACTION_1'))
    for qp in act.qfcProjects:
        if qp.uid == uid:
            return act.get_caps(qp)
    raise ValueError(f'{uid!r} not found')


@u.fixture(scope='module')
def root():
    tu.create_tables()
    yield _root(tu.qgs_path('caps'))


##


def test_offline_postgres_layer(root: gws.Root):
    cs = _caps(root)
    le = cs.layerMap['poi_L1']

    assert le.action == caps.LayerAction.edit
    assert le.readOnly is not True
    assert le.modelEntry.gpName == 'qm_qfc_poi'
    assert le.modelEntry.tableName == 'qfc.poi'


def test_layers_sharing_a_table_share_a_model(root: gws.Root):
    cs = _caps(root)
    assert cs.layerMap['district_L2'].modelEntry is cs.modelMap['qm_qfc_district']
    assert cs.layerMap['basemap_L9'].action == caps.LayerAction.baseMap


def test_locked_layer_is_read_only(root: gws.Root):
    cs = _caps(root)
    le = cs.layerMap['district_L2']

    assert le.action == caps.LayerAction.edit
    assert le.readOnly is True


def test_sql_filter_is_extracted(root: gws.Root):
    cs = _caps(root)
    assert cs.layerMap['note_L3'].sqlFilter == '"kind" = \'a\''


def test_layer_marked_remove(root: gws.Root):
    cs = _caps(root)
    assert cs.layerMap['removed_L7'].action == caps.LayerAction.remove


def test_offline_non_postgres_layer_is_removed(root: gws.Root):
    cs = _caps(root)
    assert cs.layerMap['geojson_L5'].action == caps.LayerAction.remove


def test_subquery_table_is_removed(root: gws.Root):
    cs = _caps(root)
    assert cs.layerMap['query_L4'].action == caps.LayerAction.remove


def test_missing_table_is_removed(root: gws.Root):
    cs = _caps(root)
    assert cs.layerMap['missing_L8'].action == caps.LayerAction.remove


def test_no_action_layer_has_no_entry(root: gws.Root):
    cs = _caps(root)
    assert 'wms_L6' not in cs.layerMap


def test_base_map_single_layer(root: gws.Root):
    cs = _caps(root)
    assert cs.baseMapLayerIds == ['basemap_L9']


def test_area_of_interest(root: gws.Root):
    cs = _caps(root)
    u.check.close(cs.areaOfInterest.extent, (744000, 6644000, 754000, 6654000))
    assert cs.areaOfInterest.crs.srid == 3857
    assert cs.copyOnlyAreaOfInterest is False


def test_copy_dirs(root: gws.Root):
    cs = _caps(root)
    d = tu.WORK_DIR
    # DCIM/sub is nested in DCIM and must be dropped
    assert cs.copyDirs == [f'{d}/DCIM', '/tmp/qfc_abs']


def test_path_props_for_edit_layers(root: gws.Root):
    cs = _caps(root)

    le = cs.layerMap['poi_L1']
    assert le.dataSourceFileName == 'qm_qfc_poi.gpkg'
    assert le.dataSource == './qm_qfc_poi.gpkg|layername=qm_qfc_poi'
    assert le.dataProvider == 'ogr'

    le = cs.layerMap['note_L3']
    assert le.dataSource == './qm_qfc_note.gpkg|layername=qm_qfc_note|subset="kind" = \'a\''


def test_path_props_for_base_map_layers(root: gws.Root):
    cs = _caps(root)
    le = cs.layerMap['basemap_L9']

    assert le.dataSourceFileName == gws.u.to_uid('basemap_L9') + '.gpkg'
    assert le.dataSource == './' + le.dataSourceFileName
    assert le.dataProvider == 'gdal'


##


def test_caps_are_cached(root: gws.Root):
    act = cast(action_mod.Object, root.get('ACTION_1'))
    qp = act.qfcProjects[0]

    cs1 = act.get_caps(qp)
    cs2 = act.get_caps(qp)
    assert cs1 is cs2


def test_caps_cache_is_invalidated_when_the_project_changes(root: gws.Root):
    act = cast(action_mod.Object, root.get('ACTION_1'))
    qp = act.qfcProjects[0]

    cs1 = act.get_caps(qp)

    path = qp.qgisProvider.store.path
    gws.u.write_file(path, gws.u.read_file(path).replace('<title>qfield_test</title>', '<title>changed</title>'))

    cs2 = act.get_caps(qp)
    assert cs2 is not cs1
    assert cs2.sourceHash != cs1.sourceHash

    gws.u.write_file(path, gws.u.read_file(path).replace('<title>changed</title>', '<title>qfield_test</title>'))


##
# configured models


_CONFIGURED_MODELS = """
    models+ {
        uid "MODEL_POI"
        type "postgres"
        tableName "qfc.poi"
        isEditable true
        permissions.edit "allow all"
        fields+ { name "id"   type "integer" }
        fields+ { name "name" type "text" }
        fields+ { name "geom" type "geometry" }
    }
    models+ {
        uid "MODEL_DISTRICT"
        type "postgres"
        tableName "qfc.district"
        isEditable false
    }
"""


def test_configured_model_is_used():
    root = _root(tu.qgs_path('caps_models'), _CONFIGURED_MODELS)
    cs = _caps(root)

    le = cs.layerMap['poi_L1']
    assert le.action == caps.LayerAction.edit
    assert le.modelEntry.model.uid == 'MODEL_POI'


def test_read_only_layer_may_use_a_non_editable_model():
    root = _root(tu.qgs_path('caps_models'), _CONFIGURED_MODELS)
    cs = _caps(root)

    # district_L2 is fully locked, so a non-editable model is acceptable
    le = cs.layerMap['district_L2']
    assert le.action == caps.LayerAction.edit
    assert le.modelEntry.model.uid == 'MODEL_DISTRICT'


def test_writable_layer_with_a_non_editable_model_is_removed():
    def patch(root_el):
        for name in ('is_attribute_editing_locked', 'is_feature_addition_locked', 'is_feature_deletion_locked', 'is_geometry_editing_locked'):
            tu.set_layer_prop(root_el, 'district_L2', name, 'false', 'bool')

    root = _root(tu.qgs_path('caps_unlocked', patch), _CONFIGURED_MODELS)
    cs = _caps(root)

    assert cs.layerMap['district_L2'].action == caps.LayerAction.remove


##
# project property variants


def test_base_map_from_map_theme():
    def patch(root_el):
        tu.set_project_prop(root_el, 'baseMapType', 'mapTheme')

    root = _root(tu.qgs_path('caps_theme', patch))
    cs = _caps(root)

    assert cs.baseMapLayerIds == ['wms_L6']
    assert cs.layerMap['wms_L6'].action == caps.LayerAction.baseMap


def test_unknown_map_theme_yields_no_base_map():
    def patch(root_el):
        tu.set_project_prop(root_el, 'baseMapType', 'mapTheme')
        tu.set_project_prop(root_el, 'baseMapTheme', 'NO_SUCH_THEME')

    root = _root(tu.qgs_path('caps_no_theme', patch))
    cs = _caps(root)

    assert cs.baseMapLayerIds == []


def test_base_map_disabled():
    def patch(root_el):
        tu.set_project_prop(root_el, 'createBaseMap', '0', 'int')

    root = _root(tu.qgs_path('caps_no_base_map', patch))
    cs = _caps(root)

    assert cs.baseMapLayerIds == []
    assert 'basemap_L9' not in cs.layerMap


def test_copy_only_area_of_interest():
    def patch(root_el):
        tu.set_project_prop(root_el, 'offlineCopyOnlyAoi', '1', 'int')

    root = _root(tu.qgs_path('caps_aoi', patch))
    cs = _caps(root)

    assert cs.copyOnlyAreaOfInterest is True
