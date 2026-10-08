import json
import re

import gws
import gws.lib.jsonx
import gws.lib.sa as sa
import gws.lib.vendor.umsgpack as umsgpack
import gws.test.util as u

from gws.plugin.alkis.data import indexer
from . import data

INDEX_SCHEMA_C = data.SCHEMA + '_c'


def _action_cfg(uid, index_schema, extra=''):
    return f'''
        actions+ {{
            uid "{uid}"
            type "alkis"
            dbUid "GWS_TEST_POSTGRES_PROVIDER"
            dataSchema "{data.SCHEMA}"
            indexSchema "{index_schema}"
            crs {data.SRID}
            limit 100
            {extra}
        }}
    '''


def _project_cfg(uid, action_cfg):
    return f'''
        projects+ {{
            uid "{uid}"
            map {{
                crs {data.SRID}
                layers+ {{ type "postgres" tableName "{data.SCHEMA}.ax_flurstueck" }}
            }}
            {action_cfg}
        }}
    '''


def create_tables(tables: dict):
    """Create the source tables in the test database.

    Args:
        tables: Tables as returned by ``data.load``.
    """

    u.pg.create_schema(data.SCHEMA)

    for name, tab in tables.items():
        table_name = f'{data.SCHEMA}.{name}'
        u.pg.create(table_name, {'ogc_fid': 'serial primary key', **tab['columns']})

        exprs = []
        for col, typ in tab['columns'].items():
            m = re.match(r'geometry\(\w+,\s*(\d+)\)', typ)
            exprs.append(f'ST_GeomFromText(:{col}, {m.group(1)})' if m else f'CAST(:{col} AS {typ})')
        sql = f'INSERT INTO {table_name} ({", ".join(tab["columns"])}) VALUES ({", ".join(exprs)})'

        with u.pg.connect() as conn:
            conn.execute(sa.text(sql), tab['rows'])
            conn.commit()


def create_root() -> gws.Root:
    """Configure the projects and build the indexes from the source tables."""

    u.pg.create_schema(INDEX_SCHEMA_C)

    export_cfg = '''
        exporters+ {
            uid "EXPORT"
            type "geojson"
            models+ {
                fields [
                    { type "text" name "fs_uid" title "uid" }
                    { type "text" name "fs_nutzungList_name_text" title "nutzung" }
                    { type "text" name "fs_festlegungList_name_text" title "festlegung" }
                    { type "text" name "fs_bewertungList_name_text" title "bewertung" }
                ]
            }
        }
        exporters+ {
            uid "EXPORT_PROPS"
            type "geojson"
            models+ {
                fields [
                    { type "text" name "fs_uid" title "uid" }
                    { type "text" name "fs_nutzungList_recs_props_funktion_text" title "nutzung_funktion" }
                    { type "text" name "fs_bewertungList_recs_props_nutzungsart_text" title "bewertung_nutzungsart" }
                    { type "text" name "fs_bewertungList_recs_props_bodenstufe_text" title "bewertung_bodenstufe" }
                    { type "text" name "fs_bewertungList_recs_props_klimastufe_text" title "bewertung_klimastufe" }
                    { type "text" name "fs_bewertungList_recs_props_entstehungsart_text" title "bewertung_entstehungsart" }
                    { type "text" name "fs_gebaeudeList_recs_props_objekthoehe" title "gebaeude_objekthoehe" }
                    { type "text" name "fs_gebaeudeList_recs_props_anzahlDerOberirdischenGeschosse" title "gebaeude_geschosse" }
                ]
            }
        }
    '''
    access_cfg = '''
        eigentuemer.access "allow all"
        buchung.access "allow all"
    '''

    cfg = '\n'.join([
        'permissions.all "allow all"',
        _project_cfg('A', _action_cfg('ACTION_A', data.SCHEMA, access_cfg + export_cfg)),
        _project_cfg('B', _action_cfg('ACTION_B', data.SCHEMA)),
        _project_cfg('D', _action_cfg('ACTION_D', data.SCHEMA, 'eigentuemer.access "allow all"')),
        _project_cfg('C', _action_cfg(
            'ACTION_C',
            INDEX_SCHEMA_C,
            f'''
                excludeGemarkung ["{data.GEMARKUNG_3}"]
                gemarkungFilter ["{data.LAND + data.GEMARKUNG_1}", "{data.LAND + data.GEMARKUNG_3}"]
            ''',
        )),
    ])

    root = u.gws_root(cfg)

    for uid in 'ACTION_A', 'ACTION_C':
        indexer.run(root.get(uid).ix, data.SCHEMA, with_force=True)

    return root


def run_case(root: gws.Root, case: dict) -> tuple[int, object]:
    """Send the request of a test case and extract the part of the response selected by the view.

    Args:
        root: Root object.
        case: Test case with ``command``, ``request`` and ``view``.

    Returns:
        The HTTP status and the extracted response. The response is ``None``
        if the status is not 200 or the view is ``None``.
    """

    view = case['view']
    request = dict(projectUid='A') | case['request']
    headers = {'accept': 'application/msgpack'} if case['command'] == 'alkisExportFlurstueck' else {}

    res = u.http.api(root, case['command'], request, headers=headers)
    if res.status_code != 200 or view is None:
        return res.status_code, None

    if view == 'uids':
        return 200, sorted(f['attributes']['uid'] for f in res.json['features'])

    if view == 'geometry':
        return 200, _round(res.json['features'][0]['attributes']['geometry'])

    if view == 'export':
        content = umsgpack.loads(res.data)['content']
        features = gws.lib.jsonx.from_string(content.decode('utf8'))['features']
        return 200, sorted((f['properties'] for f in features), key=lambda p: p['uid'])

    if view == 'toponyms':
        return 200, {k: res.json[k] for k in ('gemeinde', 'gemarkung', 'strasse')}

    raise ValueError(f'unknown view {view!r}')


def _round(v):
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_round(x) for x in v]
    if isinstance(v, float):
        v = round(v, 2 if abs(v) > 1000 else 6)
        return int(v) if v.is_integer() else v
    return v


##


def pytest_generate_tests(metafunc):
    if 'case' in metafunc.fixturenames:
        cases = data.load()['cases']
        ids = [f'{n:03d} {c["command"]} {json.dumps(c["request"], ensure_ascii=False)}' for n, c in enumerate(cases, 1)]
        metafunc.parametrize('case', cases, ids=ids)


@u.fixture(scope='module')
def root():
    create_tables(data.load()['tables'])
    yield create_root()


def test_case(root, case):
    assert run_case(root, case) == (case['status'], case['response'])
