"""Unit tests for qfieldcloud, no services required."""

import hashlib
import io

import gws
import gws.lib.datetimex as dtx
import gws.lib.xmlx
import gws.test.util as u

from .. import action, caps, packager

PART_SIZE = 8 * 1024 * 1024


##
# action._get_md5sum_file


def test_md5sum_small_file():
    b = b'hello world' * 100
    assert action._get_md5sum_file(io.BytesIO(b)) == hashlib.md5(b).hexdigest()


def test_md5sum_exactly_one_part():
    b = b'x' * PART_SIZE
    assert action._get_md5sum_file(io.BytesIO(b)) == hashlib.md5(b).hexdigest()


def test_md5sum_multipart():
    b = b'x' * PART_SIZE + b'y' * (1024 * 1024)

    sums = hashlib.md5(b[:PART_SIZE]).digest() + hashlib.md5(b[PART_SIZE:]).digest()
    expected = hashlib.md5(sums).hexdigest() + '-2'

    assert action._get_md5sum_file(io.BytesIO(b)) == expected


def test_md5sum_multipart_exact_multiple():
    b = b'x' * PART_SIZE + b'y' * PART_SIZE

    sums = hashlib.md5(b[:PART_SIZE]).digest() + hashlib.md5(b[PART_SIZE:]).digest()
    expected = hashlib.md5(sums).hexdigest() + '-2'

    assert action._get_md5sum_file(io.BytesIO(b)) == expected


##
# action._format_files


def test_format_files(tmp_path):
    p = tmp_path / 'a.txt'
    p.write_bytes(b'12345')

    fs = action._format_files({'a.txt': str(p)})

    assert len(fs) == 1
    assert fs[0].name == 'a.txt'
    assert fs[0].size == 5
    assert fs[0].is_attachment is False
    assert fs[0].md5sum == hashlib.md5(b'12345').hexdigest()
    assert fs[0].sha256 == hashlib.sha256(b'12345').hexdigest()
    assert fs[0].uploaded_at == fs[0].last_modified


def test_format_files_empty():
    assert action._format_files({}) == []


##
# action._format_job


def _job(state, payload=None):
    return gws.Job(
        uid='JOB_UID',
        state=state,
        payload=payload or {},
        timeCreated=dtx.parse('2026-01-02T03:04:05'),
        timeUpdated=dtx.parse('2026-01-02T03:04:06'),
    )


def test_format_job_state_mapping():
    m = {
        gws.JobState.open: 'pending',
        gws.JobState.running: 'started',
        gws.JobState.complete: 'finished',
        gws.JobState.error: 'failed',
        gws.JobState.cancel: 'pending',
    }
    for state, expected in m.items():
        assert action._format_job(_job(state), None).status == expected


def test_format_job_timestamps():
    assert action._format_job(_job(gws.JobState.open), None).started_at is None
    assert action._format_job(_job(gws.JobState.open), None).finished_at is None

    j = action._format_job(_job(gws.JobState.running), None)
    assert j.started_at is not None
    assert j.finished_at is None

    j = action._format_job(_job(gws.JobState.complete), None)
    assert j.finished_at is not None


def test_format_job_reads_the_worker_payload():
    payload = gws.u.to_dict(
        action.WorkerPayload(
            actionUid='ACTION',
            jobType='package',
            qfcProjectUid='QFC_1',
            projectUid='PROJECT_1',
        )
    )
    j = action._format_job(_job(gws.JobState.open, payload), None)

    assert j.id == 'JOB_UID'
    assert j.type == 'package'
    assert j.project_id == 'QFC_1'


##
# caps._dict_to_data


class _Target(gws.Data):
    b: bool
    i: int
    f: float
    s: str
    d: dict
    ls: list[str]


def test_dict_to_data_types():
    t = caps._dict_to_data(
        {'b': 1, 'i': '42', 'f': '1.5', 's': 'text', 'd': '{"a": 1}', 'ls': ['x', 'y']},
        _Target(),
    )
    assert t.b is True
    assert t.i == 42
    assert t.f == 1.5
    assert t.s == 'text'
    assert t.d == {'a': 1}
    assert t.ls == ['x', 'y']


def test_dict_to_data_bool_from_qgis_types():
    # QFieldSync writes flags both as `type="int"` (1/0) and as `type="bool"` (true/false),
    # the qgis parser returns an int in the first case and a real bool in the second
    assert caps._dict_to_data({'b': 1}, _Target()).b is True
    assert caps._dict_to_data({'b': '1'}, _Target()).b is True
    assert caps._dict_to_data({'b': True}, _Target()).b is True

    assert caps._dict_to_data({'b': 0}, _Target()).b is False
    assert caps._dict_to_data({'b': False}, _Target()).b is False


def test_dict_to_data_skips_missing_and_invalid():
    t = caps._dict_to_data({'i': 'not a number', 'd': 'not json'}, _Target())
    assert t.i is None
    assert t.d is None
    assert t.s is None


##
# caps.Parser.gp_name_for_model


def test_gp_name_for_model():
    pa = caps.Parser.__new__(caps.Parser)
    assert pa.gp_name_for_model('poi') == 'qm_public_poi'
    assert pa.gp_name_for_model('edit.poi') == 'qm_edit_poi'
    assert pa.gp_name_for_model('Edit.POI') == 'qm_edit_poi'


##
# caps.Parser.parse_copy_dirs


def _copy_dirs(props: dict, qgis_path: str = ''):
    pa = caps.Parser.__new__(caps.Parser)
    pa.caps = caps.Caps(
        qgisPath=qgis_path,
        copyDirs=[],
        projectProps=caps._dict_to_data(props, caps.ProjectProps()),
    )
    pa.parse_copy_dirs()
    return pa.caps.copyDirs


@u.fixture
def qgis_path(tmp_path):
    # relative dirs are resolved against the directory of the project file
    p = tmp_path / 'main.qgs'
    p.write_text('')
    return str(p)


def test_copy_dirs_absolute_and_relative(qgis_path, tmp_path):
    ds = _copy_dirs({'attachmentDirs': ['DCIM', '/abs/media']}, qgis_path)
    assert ds == ['/abs/media', f'{tmp_path}/DCIM']


def test_copy_dirs_from_dirs_to_copy(qgis_path, tmp_path):
    ds = _copy_dirs({'dirsToCopy': '{"yes": true, "no": false}'}, qgis_path)
    assert ds == [f'{tmp_path}/yes']


def test_copy_dirs_are_unnested():
    ds = _copy_dirs({'dataDirs': ['/a', '/a/b', '/a/b/c', '/z']})
    assert ds == ['/a', '/z']


def test_copy_dirs_relative_without_qgis_path_are_dropped():
    ds = _copy_dirs({'attachmentDirs': ['DCIM', '/abs/media']}, qgis_path='')
    assert ds == ['/abs/media']


##
# packager.Object.replace_vars


def test_replace_vars():
    po = packager.Object()
    po.user = gws.Data(authToken='TOKEN', loginName='LOGIN', displayName='DISPLAY')

    s = po.replace_vars('a {user.authToken} b {user.loginName} c {user.displayName} d')
    assert s == 'a TOKEN b LOGIN c DISPLAY d'


##
# packager.QgisXmlTransformer


def _entry(action_, **kwargs):
    return caps.LayerEntry(
        action=action_,
        dataSource=kwargs.get('dataSource', './x.gpkg|layername=x'),
        dataProvider=kwargs.get('dataProvider', 'ogr'),
    )


def _transform(xml: str, layer_map: dict) -> gws.XmlElement:
    po = gws.Data(uid='TEST', caps=caps.Caps(layerMap=layer_map))
    root_el = gws.lib.xmlx.from_string(xml)
    packager.QgisXmlTransformer().run(po, root_el)
    return root_el


_EDIT = _entry(caps.LayerAction.edit, dataSource='./qm_a.gpkg|layername=qm_a', dataProvider='ogr')
_BASE = _entry(caps.LayerAction.baseMap, dataSource='./b.gpkg', dataProvider='gdal')
_REMOVE = caps.LayerEntry(action=caps.LayerAction.remove)


def test_transformer_rewrites_layer_tree():
    root_el = _transform(
        """
            <qgis>
                <layer-tree-group>
                    <layer-tree-layer id="A" source="pg" providerKey="postgres"/>
                    <layer-tree-layer id="UNKNOWN" source="keep" providerKey="keep"/>
                </layer-tree-group>
            </qgis>
        """,
        {'A': _EDIT},
    )

    els = root_el.findall('.//layer-tree-layer')
    assert [e.get('id') for e in els] == ['A', 'UNKNOWN']
    assert els[0].get('source') == './qm_a.gpkg|layername=qm_a'
    assert els[0].get('providerKey') == 'ogr'
    assert els[1].get('source') == 'keep'


def test_transformer_rewrites_map_layers():
    root_el = _transform(
        """
            <qgis>
                <layer-tree-group>
                    <layer-tree-layer id="A"/>
                </layer-tree-group>
                <projectlayers>
                    <maplayer><id>A</id><datasource>pg</datasource><provider>postgres</provider></maplayer>
                    <maplayer><id>UNKNOWN</id><datasource>keep</datasource><provider>keep</provider></maplayer>
                </projectlayers>
            </qgis>
        """,
        {'A': _BASE},
    )

    els = root_el.findall('.//maplayer')
    assert els[0].textof('datasource') == './b.gpkg'
    assert els[0].textof('provider') == 'gdal'
    assert els[1].textof('datasource') == 'keep'


def test_transformer_removes_layers():
    root_el = _transform(
        """
            <qgis>
                <layer-tree-group>
                    <layer-tree-layer id="A"/>
                    <layer-tree-layer id="B"/>
                </layer-tree-group>
                <projectlayers>
                    <maplayer><id>A</id><datasource>x</datasource><provider>x</provider></maplayer>
                    <maplayer><id>B</id><datasource>x</datasource><provider>x</provider></maplayer>
                </projectlayers>
            </qgis>
        """,
        {'A': _EDIT, 'B': _REMOVE},
    )

    assert [e.get('id') for e in root_el.findall('.//layer-tree-layer')] == ['A']
    assert [e.textof('id') for e in root_el.findall('.//maplayer')] == ['A']


def test_transformer_drops_empty_groups():
    root_el = _transform(
        """
            <qgis>
                <layer-tree-group>
                    <layer-tree-group name="EMPTY">
                        <layer-tree-layer id="B"/>
                    </layer-tree-group>
                    <layer-tree-group name="MIXED">
                        <layer-tree-layer id="A"/>
                        <layer-tree-layer id="B"/>
                    </layer-tree-group>
                </layer-tree-group>
            </qgis>
        """,
        {'A': _EDIT, 'B': _REMOVE},
    )

    names = [e.get('name') for e in root_el.findall('layer-tree-group/layer-tree-group')]
    assert names == ['MIXED']


def test_transformer_drops_nested_empty_groups():
    root_el = _transform(
        """
            <qgis>
                <layer-tree-group>
                    <layer-tree-group name="OUTER">
                        <layer-tree-group name="INNER">
                            <layer-tree-layer id="B"/>
                        </layer-tree-group>
                    </layer-tree-group>
                </layer-tree-group>
            </qgis>
        """,
        {'B': _REMOVE},
    )

    assert root_el.findall('.//layer-tree-group') == []


def test_transformer_rewrites_referenced_layers():
    root_el = _transform(
        """
            <qgis>
                <layer-tree-group/>
                <projectlayers>
                    <maplayer>
                        <id>X</id>
                        <referencedLayers>
                            <relation referencedLayer="A" providerKey="postgres" dataSource="pg"/>
                            <relation referencedLayer="B" providerKey="postgres" dataSource="pg"/>
                            <relation referencedLayer="A"/>
                        </referencedLayers>
                    </maplayer>
                </projectlayers>
            </qgis>
        """,
        {'A': _EDIT, 'B': _REMOVE},
    )

    els = root_el.findall('.//referencedLayers/relation')

    assert els[0].get('dataSource') == './qm_a.gpkg|layername=qm_a'
    assert els[0].get('providerKey') == 'ogr'

    # not an edit layer, left alone
    assert els[1].get('dataSource') == 'pg'

    # attributes not present, not added
    assert 'dataSource' not in els[2].attrib
    assert 'providerKey' not in els[2].attrib


def test_transformer_rewrites_edit_widgets():
    root_el = _transform(
        """
            <qgis>
                <layer-tree-group/>
                <projectlayers>
                    <maplayer>
                        <id>X</id>
                        <fieldConfiguration>
                            <field name="f1">
                                <editWidget type="RelationReference">
                                    <config><Option type="Map">
                                        <Option value="A" name="ReferencedLayerId" type="QString"/>
                                        <Option value="old" name="ReferencedLayerDataSource" type="QString"/>
                                        <Option value="old" name="ReferencedLayerProviderKey" type="QString"/>
                                    </Option></config>
                                </editWidget>
                            </field>
                            <field name="f2">
                                <editWidget type="TextEdit">
                                    <config><Option type="Map">
                                        <Option value="A" name="ReferencedLayerId" type="QString"/>
                                        <Option value="old" name="ReferencedLayerDataSource" type="QString"/>
                                    </Option></config>
                                </editWidget>
                            </field>
                            <field name="f3">
                                <editWidget type="RelationReference">
                                    <config><Option type="Map">
                                        <Option value="B" name="ReferencedLayerId" type="QString"/>
                                        <Option value="old" name="ReferencedLayerDataSource" type="QString"/>
                                    </Option></config>
                                </editWidget>
                            </field>
                        </fieldConfiguration>
                    </maplayer>
                </projectlayers>
            </qgis>
        """,
        {'A': _EDIT, 'B': _REMOVE},
    )

    def _opt(field, name):
        for el in root_el.findall('.//field'):
            if el.get('name') == field:
                for o in el.findall('.//Option'):
                    if o.get('name') == name:
                        return o.get('value')

    assert _opt('f1', 'ReferencedLayerDataSource') == './qm_a.gpkg|layername=qm_a'
    assert _opt('f1', 'ReferencedLayerProviderKey') == 'ogr'

    # not a RelationReference widget
    assert _opt('f2', 'ReferencedLayerDataSource') == 'old'

    # not an edit layer
    assert _opt('f3', 'ReferencedLayerDataSource') == 'old'


def test_transformer_forces_relative_paths():
    root_el = _transform(
        """
            <qgis>
                <layer-tree-group/>
                <properties><Paths><Absolute type="bool">true</Absolute></Paths></properties>
            </qgis>
        """,
        {},
    )
    assert root_el.textof('properties/Paths/Absolute') == 'false'


def test_transformer_adds_relative_paths_property():
    root_el = _transform('<qgis><layer-tree-group/></qgis>', {})
    assert root_el.textof('properties/Paths/Absolute') == 'false'
    assert root_el.require('properties/Paths/Absolute').get('type') == 'bool'
