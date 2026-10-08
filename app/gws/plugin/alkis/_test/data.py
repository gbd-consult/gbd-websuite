"""ALKIS test dataset.

Creates a small ALKIS dataset in the norBIT table layout (GeoInfoDok 6),
as read by ``gws.plugin.alkis.data.norbit6``. Table and column definitions
follow the norBIT import, only the columns used by the reader are created.

Most objects have two versions: a historic one (``endet`` set) and a current
one. Some objects only have an ended version.

Places:

- Land1 > Regierungsbezirk1 > Kreis1 > Gemeinde1, Gemeinde2
- Gemeinde3 has ended, Flurstuecke in it are excluded
- Gemarkung1, Gemarkung2 (in Gemeinde1, Gemeinde2), Gemarkung3 (for ``excludeGemarkung``)
- Buchungsblattbezirk1, Dienststelle1

Flurstuecke (100 x 100 m squares, in a row along the x axis):

- fs 1: Gemarkung1, booked on bs 1 (was bs 5), two addresses, one Lage without number, two buildings
- fs 2: Gemarkung1, addresses for house number ranges, booked on bs 3, which is booked ``zu`` bs 2, booked ``an`` bs 1
- fs 3: Gemarkung2, Gemeinde2, unencoded addresses, with nenner and flurstuecksfolge
- fs 4: ended, booked on the ended bs 6
- fs 5: Gemeinde3 (ended), excluded
- fs 6: no geometry
- fs 7: Gemarkung3
- hfs 1: ``AX_HistorischesFlurstueck`` with Buchung data, successors fs 1, fs 2 and an unknown one
- hfs 2: ``AX_HistorischesFlurstueck`` without Buchung data

Land register:

- bb 1: bs 1, bs 4, bs 5 (ended); nn 1 (Person1), nn 2 (Firma1), nn 3 (ended, Person3)
- bb 2: bs 2 (Erbbaurecht ``an`` bs 1); nn 4 (Person4)
- bb 3: bs 3 (``zu`` bs 2); nn 5 (Erbengemeinschaft), nn 6 (Person1)
- bb 4: ended; bs 6 (ended)

Parts: a Wohnbauflaeche over fs 1 and fs 2, Strassenverkehr over fs 3 (and an
ended one), a Bodenschaetzung over half of fs 3, a Denkmalschutzrecht over a
quarter of fs 1 that touches fs 2 by less than ``MIN_PART_AREA``.

The dataset and the test cases with their expected results are kept in
``data.md``, which drives the tests. ``generate`` creates ``data.md`` from the
dataset and the requests in ``_TESTS``, by running the requests against the
test database; review the expected results after generating. ``load`` reads
``data.md``. To generate, run in the test container::

    python3 /gws-app/gws/plugin/alkis/_test/data.py <base dir> > data.md

where ``<base dir>`` is ``runner.base_dir`` of the test harness.
"""

import contextlib
import json
import os
import sys

import gws
import gws.test.util as u

SCHEMA = 'alkis_test'

SRID = 25832

X0 = 400000
Y0 = 5700000

BEGIN_OLD = '2020-01-01T00:00:00Z'
"""Start of historic versions."""
BEGIN_NEW = '2025-06-01T00:00:00Z'
"""End of historic versions, start of current versions."""

ANLASS_OLD = '010101'
ANLASS_NEW = '010102'
ANLASS_END = '300500'
ANLASS_NONE = '000000'

LAND = '05'
REGIERUNGSBEZIRK = '9'
KREIS = '78'
GEMEINDE_1 = '001'
GEMEINDE_2 = '002'
GEMEINDE_3 = '003'
GEMARKUNG_1 = '1001'
GEMARKUNG_2 = '1002'
GEMARKUNG_3 = '1003'
BEZIRK = '2001'
STELLE = '3001'
LAGE_1 = '00001'
LAGE_2 = '00002'

HISTORY = dict(wantHistorySearch=True, wantHistoryDisplay=True)

MD_PATH = os.path.dirname(__file__) + '/data.md'


def _uid(kind: str, n: int) -> str:
    """Return the 16-character ``gml_id`` of a test object, e.g. ``_uid('fs', 1)`` -> ``DEFS000000000001``."""

    s = 'DE' + kind.upper()
    return s + str(n).zfill(16 - len(s))


def _flurstueckskennzeichen(gemarkung: str, flur: int, zaehler: int, nenner: int = 0, folge: str = '') -> str:
    """Return a Flurstueckskennzeichen in the ALKIS format."""

    return (
            LAND
            + gemarkung
            + f'{flur:03d}'
            + f'{zaehler:05d}'
            + (f'{nenner:04d}' if nenner else '____')
            + (folge or '__')
    )


def _buchungsblattkennzeichen(n: int) -> str:
    """Return the Buchungsblattkennzeichen of ``bb n``."""

    return LAND + BEZIRK + f'{n:06d}'


def generate(base_dir: str):
    """Create the dataset in the test database, run the test requests and print ``data.md`` to stdout.

    Args:
        base_dir: ``runner.base_dir`` of the test harness.
    """

    from gws.plugin.alkis._test import action_test

    u.load_options(base_dir)
    gws.u.ensure_system_dirs()
    gws.log.set_level('ERROR')

    with contextlib.redirect_stdout(sys.stderr):
        tables = _tables_json()
        action_test.create_tables(tables)
        root = action_test.create_root()
        lines = [
            *_render_tables(tables),
            *_render_tests(root, action_test.run_case),
        ]

    sys.stdout.write('\n'.join(lines).rstrip() + '\n')


def load(path: str = MD_PATH) -> dict:
    """Read ``data.md``.

    Args:
        path: Path to the markdown file.

    Returns:
        A dict with the keys ``tables`` and ``cases``. ``tables`` maps a table
        name to ``{columns: {name: type}, rows: [{name: value}]}``, values are
        strings or ``None``. ``cases`` is a list of
        ``{section, group, command, request, status, view, response}``;
        ``view`` and ``response`` are ``None`` if the status is not 200.
    """

    tables = {}
    cases = []

    part = ''
    section = ''
    group = ''
    table = None
    block = None

    for line in gws.u.read_file(path).splitlines():
        if block is not None:
            if line.startswith('```'):
                for text in '\n'.join(block).strip().split('\n\n'):
                    cases.append(_parse_case(text, section, group))
                block = None
            else:
                block.append(line)
            continue

        if line.startswith('# '):
            part = line[2:].strip()
            continue
        if line.startswith('## '):
            section = line[3:].strip()
            group = ''
            if part == 'Source data':
                name = section.removeprefix('Table ').strip()
                table = tables[name] = dict(columns={}, rows=[])
            continue
        if line.startswith('### '):
            group = line[4:].strip()
            continue
        if line.startswith('```'):
            block = []
            continue

        if part == 'Source data' and table is not None and line.startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if not table['columns']:
                for c in cells:
                    name, _, typ = c.partition(' ')
                    table['columns'][name] = typ.strip()
            elif not set(''.join(cells)) <= set('-'):
                table['rows'].append({name: c or None for name, c in zip(table['columns'], cells)})

    return dict(tables=tables, cases=cases)


##

def _point_props(x, y):
    return {'crs': f'EPSG:{SRID}', 'geometry': {'type': 'Point', 'coordinates': [X0 + x, Y0 + y]}}


_TESTS = [
    dict(
        title='Find Flurstueck',
        command='alkisFindFlurstueck',
        view='uids',
        groups=[
            ('Places', [
                dict(gemarkungCode=GEMARKUNG_1),
                dict(gemarkungCode=LAND + GEMARKUNG_2),
                dict(gemarkungCode=GEMARKUNG_3),
                dict(gemarkung='Gemarkung1'),
                dict(gemarkung='gemarkung1'),
                dict(gemeinde='Gemeinde2'),
                dict(gemeindeCode=LAND + REGIERUNGSBEZIRK + KREIS + GEMEINDE_1),
                dict(kreis='Kreis1'),
                dict(regierungsbezirk='Regierungsbezirk1'),
                dict(land='Land1'),
                dict(landCode=LAND, gemarkungCode=GEMARKUNG_1),
            ]),
            ('Parcel numbers', [
                dict(flurnummer='2'),
                dict(zaehler='1'),
                dict(zaehler='2', nenner='1'),
                dict(flurstuecksfolge='01'),
                dict(fsnummer=LAND + GEMARKUNG_1 + ' 1-2/1'),
                dict(fsnummer='1-1'),
                dict(fsnummer=LAND + GEMARKUNG_2 + ' 2-3/2 (01)'),
                dict(fsnummer=_flurstueckskennzeichen(GEMARKUNG_2, 2, 3, 2, '01')),
                dict(fsnummer=_flurstueckskennzeichen(GEMARKUNG_1, 1, 1)[:14]),
                dict(fsnummer=_uid('fs', 3)),
                dict(combinedFlurstueckCode=f'{LAND}_{GEMARKUNG_1}_1_2_1'),
                dict(combinedFlurstueckCode=f'0_{GEMARKUNG_1}_0_1'),
                dict(uids=[_uid('fs', 1), _uid('fs', 3)]),
            ]),
            ('Area', [
                dict(flaecheVon=10000, flaecheBis=10000),
                dict(flaecheBis=9995),
                dict(flaecheBis=9995, **HISTORY),
            ]),
            ('Search area', [
                dict(shapes=[_point_props(250, 50)]),
                dict(shapes=[_point_props(450, 50)]),
                dict(shapes=[_point_props(50, 50), _point_props(650, 50)]),
            ]),
            ('Addresses', [
                dict(strasse='Strasse1'),
                dict(strasse='strasse1'),
                dict(strasse='Str'),
                dict(strasse='Gewann1'),
                dict(strasse='Strasse1', hausnummer='1'),
                dict(strasse='Strasse1', hausnummer='*'),
                dict(strasse='Strasse2', hausnummer='10'),
                dict(strasse='Strasse1', hausnummer='1a'),
                dict(strasse='Strasse1', hausnummer='1a', **HISTORY),
                dict(strasse='Strasse1', hausnummer='9'),
                dict(strasse='Strasse3alt'),
                dict(strasse='Strasse3alt', **HISTORY),
            ]),
            ('History', [
                dict(gemarkungCode=GEMARKUNG_1, **HISTORY),
                dict(gemarkungCode=GEMARKUNG_1, wantHistorySearch=True),
                dict(gemeinde='Gemeinde1', **HISTORY),
                dict(uids=[_uid('fs', 4)]),
                dict(uids=[_uid('fs', 4)], **HISTORY),
            ]),
            ('Land register', [
                dict(bblatt=_buchungsblattkennzeichen(1)),
                dict(bblatt=_buchungsblattkennzeichen(2)),
                dict(bblatt=_buchungsblattkennzeichen(3)),
                dict(bblatt=f'{_buchungsblattkennzeichen(2)}, {_buchungsblattkennzeichen(4)}'),
                dict(bblatt=_buchungsblattkennzeichen(4)),
                dict(bblatt=_buchungsblattkennzeichen(4), **HISTORY),
            ]),
            ('Owners', [
                dict(personName='Name1'),
                dict(personName='name'),
                dict(personName='Firma1'),
                dict(personName='Name4'),
                dict(personName='Name1', personVorname='Vorname1'),
                dict(personName='Name1', personVorname='Vorname4'),
                dict(personName='Name3'),
                dict(personName='Name3', **HISTORY),
                dict(personName='Geburtsname1'),
                dict(personName='Geburtsname1', **HISTORY),
            ]),
        ],
    ),
    dict(
        title='Find Flurstueck: errors',
        command='alkisFindFlurstueck',
        view='uids',
        groups=[
            ('', [
                dict(hausnummer='1'),
                dict(personVorname='Vorname1'),
                dict(fsnummer='xyz'),
                dict(gemarkungCode=GEMARKUNG_1, limit=1),
            ]),
            ('No access to owner and land register data in project B', [
                dict(projectUid='B', personName='Name1'),
                dict(projectUid='B', bblatt=_buchungsblattkennzeichen(1)),
                dict(projectUid='B', zaehler='1', wantEigentuemer=True),
                dict(projectUid='B', zaehler='1', displayThemes=['buchung']),
                dict(projectUid='B', zaehler='1', displayThemes=['eigentuemer']),
            ]),
            ('Access to owner data, but not to land register data, in project D', [
                dict(projectUid='D', personName='Name1'),
                dict(projectUid='D', zaehler='1', wantEigentuemer=True),
                dict(projectUid='D', zaehler='1', displayThemes=['eigentuemer']),
            ]),
        ],
    ),
    dict(
        title='Find Flurstueck: other projects',
        command='alkisFindFlurstueck',
        view='uids',
        groups=[
            ('Project B: no access to owner and land register data, other searches work', [
                dict(projectUid='B', zaehler='1'),
                dict(projectUid='B', zaehler='1', displayThemes=['lage', 'gebaeude', 'nutzung']),
            ]),
            ('Project C: Gemarkung3 is excluded from the index, searches are restricted to Gemarkung1 and Gemarkung3', [
                dict(projectUid='C', kreis='Kreis1'),
                dict(projectUid='C', gemarkungCode=GEMARKUNG_2),
                dict(projectUid='C', gemarkungCode=GEMARKUNG_3),
            ]),
        ],
    ),
    dict(
        title='Find Flurstueck: geometry',
        command='alkisFindFlurstueck',
        view='geometry',
        groups=[
            ('', [
                dict(uids=[_uid('fs', 3)]),
                dict(uids=[_uid('fs', 3)], crs='EPSG:4326'),
            ]),
        ],
    ),
    dict(
        title='Export Flurstueck',
        command='alkisExportFlurstueck',
        view='export',
        text='Exporter `EXPORT` (GeoJSON) with the fields `fs_uid`, `fs_nutzungList_name_text`, `fs_festlegungList_name_text`, `fs_bewertungList_name_text`.',
        groups=[
            ('', [
                dict(exporterUid='EXPORT', findRequest=dict(
                    uids=[_uid('fs', 1), _uid('fs', 2), _uid('fs', 3)],
                    displayThemes=['nutzung', 'festlegung', 'bewertung'],
                )),
            ]),
        ],
    ),
    dict(
        title='Find Adresse',
        command='alkisFindAdresse',
        view='uids',
        groups=[
            ('', [
                dict(strasse='Strasse'),
                dict(strasse='Strasse1', hausnummer='1'),
                dict(strasse='Strasse', hausnummerNotNull=True),
                dict(strasse='Strasse1', hausnummer='*'),
            ]),
            ('House number ranges: 1, 2, 2b, 2e, 2f, 5a, 10', [
                dict(strasse='Strasse', bisHausnummer='5'),
                dict(strasse='Strasse', bisHausnummer='2a'),
                dict(strasse='Strasse', hausnummer='2', bisHausnummer='10'),
                dict(strasse='Strasse', hausnummer='2b', bisHausnummer='2e'),
                dict(strasse='Strasse', hausnummer='2 B', bisHausnummer='2 E'),
                dict(strasse='Strasse', hausnummer='2', bisHausnummer='2'),
                dict(strasse='Strasse', hausnummer='2a', bisHausnummer='5'),
                dict(strasse='Strasse', hausnummer='6', bisHausnummer='9'),
            ]),
            ('History and places', [
                dict(strasse='Strasse1', hausnummer='1a'),
                dict(strasse='Strasse1', hausnummer='1a', wantHistorySearch=True),
                dict(strasse='Strasse3', gemeinde='Gemeinde2'),
                dict(strasse='Strasse3', gemeinde='Gemeinde1'),
                dict(strasse='Gewann'),
                dict(combinedAdresseCode='Strasse1_1'),
            ]),
        ],
    ),
    dict(
        title='Find Adresse: errors',
        command='alkisFindAdresse',
        view='uids',
        groups=[
            ('', [
                dict(hausnummer='1'),
                dict(strasse='Strasse', bisHausnummer='x'),
                dict(strasse='Strasse', hausnummer='x', bisHausnummer='5'),
            ]),
        ],
    ),
    dict(
        title='Get toponyms',
        command='alkisGetToponyms',
        view='toponyms',
        groups=[
            ('', [
                dict(),
                dict(projectUid='C'),
            ]),
        ],
    ),
]
"""Test requests by section. A request without ``projectUid`` runs in project ``A``.

The view selects the part of the response that is compared, see ``action_test.run_case``.
"""


def _tables_json():
    """Convert the dataset into the ``tables`` structure returned by ``load``.

    Columns that are empty in all rows are left out.
    """

    tables = {}

    for name, (cols, rows) in _tables().items():
        cols = _COMMON_COLUMNS | cols
        cols = {c: typ for c, typ in cols.items() if any(r.get(c) is not None for r in rows)}
        tables[name] = dict(
            columns=cols,
            rows=[{c: _str(r.get(c)) for c in cols} for r in rows],
        )

    return tables


def _str(v):
    if v is None:
        return None
    if isinstance(v, list):
        return '{' + ', '.join(str(x) for x in v) + '}'
    return str(v)


def _render_tables(tables):
    yield '# Source data'
    yield ''
    yield f'Schema `{SCHEMA}`. The header cells contain the column name and the column type. Empty cells are NULL, geometries are WKT in EPSG:{SRID}.'
    yield ''

    for name, tab in tables.items():
        yield f'## Table {name}'
        yield ''

        header = [f'{c} {typ}' for c, typ in tab['columns'].items()]
        body = [[r[c] or '' for c in tab['columns']] for r in tab['rows']]
        widths = [max(len(s) for s in col) for col in zip(header, *body)]

        def line(cells):
            return '| ' + ' | '.join(s.ljust(w) for s, w in zip(cells, widths)) + ' |'

        yield line(header)
        yield '|' + '|'.join('-' * (w + 2) for w in widths) + '|'
        for cells in body:
            yield line(cells)
        yield ''


def _render_tests(root, run_case):
    yield '# Tests'
    yield ''
    yield 'A request without `projectUid` runs in project `A`. Each case is the command with the request, the HTTP status and, if the status is 200, the expected response, prefixed with the view that selects the part of the response that is compared.'
    yield ''

    for sec in _TESTS:
        yield f'## {sec["title"]}'
        yield ''
        if sec.get('text'):
            yield sec['text']
            yield ''

        for group, requests in sec['groups']:
            if group:
                yield f'### {group}'
                yield ''

            texts = []
            for request in requests:
                case = dict(view=sec['view'], command=sec['command'], request=request)
                status, response = run_case(root, case)
                lines = [f'{sec["command"]} {_compact(request)}', str(status)]
                if response is not None:
                    lines.append(sec['view'] + ' ' + _render_response(sec['view'], response))
                texts.append('\n'.join(lines))

            yield '```'
            yield '\n\n'.join(texts)
            yield '```'
            yield ''


def _render_response(view, response):
    if view == 'toponyms':
        lines = []
        for k, vals in response.items():
            lines.append(f'    "{k}": ' + _render_list(vals, indent='    '))
        return '{\n' + ',\n'.join(lines) + '\n}'
    if view == 'export':
        return _render_list(response)
    return _compact(response)


def _render_list(items, indent=''):
    if not items:
        return '[]'
    return '[\n' + ',\n'.join(indent + '    ' + _compact(v) for v in items) + '\n' + indent + ']'


def _parse_case(text, section, group):
    lines = text.split('\n')
    command, _, request = lines[0].partition(' ')
    view = response = None
    if len(lines) > 2:
        view, _, first = lines[2].partition(' ')
        response = json.loads('\n'.join([first, *lines[3:]]))
    return dict(
        section=section,
        group=group,
        command=command,
        request=json.loads(request),
        status=int(lines[1]),
        view=view,
        response=response,
    )


def _compact(v):
    return json.dumps(v, ensure_ascii=False)

##


_COMMON_COLUMNS = {
    'gml_id': 'character(16)',
    'anlass': 'varchar[]',
    'beginnt': 'character(20)',
    'endet': 'character(20)',
}

_GEOM = f'geometry(Geometry, {SRID})'


def _changed(gml_id, old=None, **row):
    """Rows of an object with a historic version and a current version.

    The historic version has the values of ``row`` updated with ``old``.
    """

    return [
        dict(row, **(old or {}), gml_id=gml_id, beginnt=BEGIN_OLD, endet=BEGIN_NEW, anlass=[ANLASS_OLD]),
        dict(row, gml_id=gml_id, beginnt=BEGIN_NEW, endet=None, anlass=[ANLASS_NEW]),
    ]


def _ended(gml_id, **row):
    """Rows of an object that has ended."""

    return [
        dict(row, gml_id=gml_id, beginnt=BEGIN_OLD, endet=BEGIN_NEW, anlass=[ANLASS_OLD, ANLASS_END]),
    ]


def _box(x1, y1, x2, y2):
    """Polygon in local coordinates, relative to ``X0``, ``Y0``."""

    x1, y1, x2, y2 = X0 + x1, Y0 + y1, X0 + x2, Y0 + y2
    return f'POLYGON(({x1} {y1}, {x2} {y1}, {x2} {y2}, {x1} {y2}, {x1} {y1}))'


def _point(x, y):
    """Point in local coordinates, relative to ``X0``, ``Y0``."""

    return f'POINT({X0 + x} {Y0 + y})'


def _fs_box(n):
    """Geometry of ``fs n``."""

    return _box((n - 1) * 100, 0, n * 100, 100)


def _tables():
    return {
        'ax_bundesland': _ax_bundesland(),
        'ax_regierungsbezirk': _ax_regierungsbezirk(),
        'ax_kreisregion': _ax_kreisregion(),
        'ax_gemeinde': _ax_gemeinde(),
        'ax_gemarkung': _ax_gemarkung(),
        'ax_buchungsblattbezirk': _ax_buchungsblattbezirk(),
        'ax_dienststelle': _ax_dienststelle(),

        'ax_lagebezeichnungkatalogeintrag': _ax_lagebezeichnungkatalogeintrag(),
        'ax_lagebezeichnungmithausnummer': _ax_lagebezeichnungmithausnummer(),
        'ax_lagebezeichnungohnehausnummer': _ax_lagebezeichnungohnehausnummer(),
        'ap_pto': _ap_pto(),
        'ax_gebaeude': _ax_gebaeude(),

        'ax_anschrift': _ax_anschrift(),
        'ax_person': _ax_person(),
        'ax_namensnummer': _ax_namensnummer(),
        'ax_buchungsblatt': _ax_buchungsblatt(),
        'ax_buchungsstelle': _ax_buchungsstelle(),

        'ax_flurstueck': _ax_flurstueck(),
        'ax_historischesflurstueck': _ax_historischesflurstueck(),

        'ax_wohnbauflaeche': _ax_wohnbauflaeche(),
        'ax_strassenverkehr': _ax_strassenverkehr(),
        'ax_bodenschaetzung': _ax_bodenschaetzung(),
        'ax_denkmalschutzrecht': _ax_denkmalschutzrecht(),
    }


##

_PLACE_COLUMNS = {
    'bezeichnung': 'varchar',
    'schluesselgesamt': 'varchar',
}


def _ax_bundesland():
    cols = _PLACE_COLUMNS | {
        'land': 'varchar',
    }
    rows = [
        *_changed(_uid('land', 1), old=dict(bezeichnung='Land1alt'), bezeichnung='Land1', land=LAND, schluesselgesamt=LAND),
    ]
    return cols, rows


def _ax_regierungsbezirk():
    cols = _PLACE_COLUMNS | {
        'land': 'varchar',
        'regierungsbezirk': 'varchar',
    }
    rows = [
        *_changed(
            _uid('rb', 1),
            old=dict(bezeichnung='Regierungsbezirk1alt'),
            bezeichnung='Regierungsbezirk1',
            land=LAND,
            regierungsbezirk=REGIERUNGSBEZIRK,
            schluesselgesamt=LAND + REGIERUNGSBEZIRK,
        ),
    ]
    return cols, rows


def _ax_kreisregion():
    cols = _PLACE_COLUMNS | {
        'land': 'varchar',
        'regierungsbezirk': 'varchar',
        'kreis': 'varchar',
    }
    rows = [
        *_changed(
            _uid('kreis', 1),
            old=dict(bezeichnung='Kreis1alt'),
            bezeichnung='Kreis1',
            land=LAND,
            regierungsbezirk=REGIERUNGSBEZIRK,
            kreis=KREIS,
            schluesselgesamt=LAND + REGIERUNGSBEZIRK + KREIS,
        ),
    ]
    return cols, rows


def _ax_gemeinde():
    cols = _PLACE_COLUMNS | {
        'land': 'varchar',
        'regierungsbezirk': 'varchar',
        'kreis': 'varchar',
        'gemeinde': 'varchar',
    }

    def row(n, code):
        return dict(
            bezeichnung=f'Gemeinde{n}',
            land=LAND,
            regierungsbezirk=REGIERUNGSBEZIRK,
            kreis=KREIS,
            gemeinde=code,
            schluesselgesamt=LAND + REGIERUNGSBEZIRK + KREIS + code,
        )

    rows = [
        *_changed(_uid('gem', 1), old=dict(bezeichnung='Gemeinde1alt'), **row(1, GEMEINDE_1)),
        *_changed(_uid('gem', 2), old=dict(bezeichnung='Gemeinde2alt'), **row(2, GEMEINDE_2)),
        *_ended(_uid('gem', 3), **row(3, GEMEINDE_3)),
    ]
    return cols, rows


def _ax_gemarkung():
    cols = _PLACE_COLUMNS | {
        'land': 'varchar',
        'gemarkungsnummer': 'varchar',
        'gemeindezugehoerigkeit_land': 'varchar[]',
        'gemeindezugehoerigkeit_regierungsbezirk': 'varchar[]',
        'gemeindezugehoerigkeit_kreis': 'varchar[]',
        'gemeindezugehoerigkeit_gemeinde': 'varchar[]',
    }

    def row(n, code, gemeinde):
        return dict(
            bezeichnung=f'Gemarkung{n}',
            land=LAND,
            gemarkungsnummer=code,
            schluesselgesamt=LAND + code,
            gemeindezugehoerigkeit_land=[LAND],
            gemeindezugehoerigkeit_regierungsbezirk=[REGIERUNGSBEZIRK],
            gemeindezugehoerigkeit_kreis=[KREIS],
            gemeindezugehoerigkeit_gemeinde=[gemeinde],
        )

    rows = [
        *_changed(_uid('gemkg', 1), old=dict(bezeichnung='Gemarkung1alt'), **row(1, GEMARKUNG_1, GEMEINDE_1)),
        *_changed(_uid('gemkg', 2), old=dict(bezeichnung='Gemarkung2alt'), **row(2, GEMARKUNG_2, GEMEINDE_2)),
        *_changed(_uid('gemkg', 3), old=dict(bezeichnung='Gemarkung3alt'), **row(3, GEMARKUNG_3, GEMEINDE_1)),
    ]
    return cols, rows


def _ax_buchungsblattbezirk():
    cols = _PLACE_COLUMNS | {
        'land': 'varchar',
        'bezirk': 'varchar',
        'gehoertzu_land': 'varchar',
        'gehoertzu_stelle': 'varchar',
    }
    rows = [
        *_changed(
            _uid('bbbez', 1),
            old=dict(bezeichnung='Buchungsblattbezirk1alt'),
            bezeichnung='Buchungsblattbezirk1',
            land=LAND,
            bezirk=BEZIRK,
            schluesselgesamt=LAND + BEZIRK,
            gehoertzu_land=LAND,
            gehoertzu_stelle=STELLE,
        ),
    ]
    return cols, rows


def _ax_dienststelle():
    cols = _PLACE_COLUMNS | {
        'land': 'varchar',
        'stelle': 'varchar',
        'stellenart': 'integer',
    }
    rows = [
        *_changed(
            _uid('dst', 1),
            old=dict(bezeichnung='Dienststelle1alt'),
            bezeichnung='Dienststelle1',
            land=LAND,
            stelle=STELLE,
            schluesselgesamt=LAND + STELLE,
            stellenart=1100,
        ),
    ]
    return cols, rows


##

_LAGE_KEY_COLUMNS = {
    'land': 'varchar',
    'regierungsbezirk': 'varchar',
    'kreis': 'varchar',
    'gemeinde': 'varchar',
    'lage': 'varchar',
}


def _lage_key(gemeinde, lage):
    return dict(land=LAND, regierungsbezirk=REGIERUNGSBEZIRK, kreis=KREIS, gemeinde=gemeinde, lage=lage)


def _ax_lagebezeichnungkatalogeintrag():
    cols = _LAGE_KEY_COLUMNS | {
        'bezeichnung': 'varchar',
        'schluesselgesamt': 'varchar',
    }
    rows = [
        *_changed(
            _uid('kat', 1),
            old=dict(bezeichnung='Strasse1alt'),
            bezeichnung='Strasse1',
            schluesselgesamt=LAND + REGIERUNGSBEZIRK + KREIS + GEMEINDE_1 + LAGE_1,
            **_lage_key(GEMEINDE_1, LAGE_1),
        ),
        *_changed(
            _uid('kat', 2),
            old=dict(bezeichnung='Strasse2alt'),
            bezeichnung='Strasse2',
            schluesselgesamt=LAND + REGIERUNGSBEZIRK + KREIS + GEMEINDE_1 + LAGE_2,
            **_lage_key(GEMEINDE_1, LAGE_2),
        ),
    ]
    return cols, rows


def _ax_lagebezeichnungmithausnummer():
    cols = _LAGE_KEY_COLUMNS | {
        'unverschluesselt': 'varchar',
        'hausnummer': 'varchar',
        'ortsteil': 'varchar',
    }
    rows = [
        *_changed(_uid('lmh', 1), old=dict(hausnummer='1a'), hausnummer='1', **_lage_key(GEMEINDE_1, LAGE_1)),
        *_changed(_uid('lmh', 2), old=dict(hausnummer='12'), hausnummer='10', **_lage_key(GEMEINDE_1, LAGE_2)),
        *_changed(_uid('lmh', 5), hausnummer='2', **_lage_key(GEMEINDE_1, LAGE_2)),
        *_changed(_uid('lmh', 6), hausnummer='2e', **_lage_key(GEMEINDE_1, LAGE_2)),
        *_changed(_uid('lmh', 7), hausnummer='2f', **_lage_key(GEMEINDE_1, LAGE_2)),
        *_changed(_uid('lmh', 8), hausnummer='5a', **_lage_key(GEMEINDE_1, LAGE_2)),
        *_changed(_uid('lmh', 3), old=dict(unverschluesselt='Strasse3alt'), unverschluesselt='Strasse3', hausnummer='2b', ortsteil='Ortsteil1'),
        *_ended(_uid('lmh', 4), hausnummer='9', **_lage_key(GEMEINDE_1, LAGE_1)),
    ]
    return cols, rows


def _ax_lagebezeichnungohnehausnummer():
    cols = _LAGE_KEY_COLUMNS | {
        'unverschluesselt': 'varchar',
        'zusatzzurlagebezeichnung': 'varchar',
        'ortsteil': 'varchar',
    }
    rows = [
        *_changed(_uid('loh', 1), old=dict(zusatzzurlagebezeichnung='Zusatz1'), **_lage_key(GEMEINDE_1, LAGE_1)),
        *_changed(_uid('loh', 2), old=dict(unverschluesselt='Gewann1alt'), unverschluesselt='Gewann1'),
    ]
    return cols, rows


def _ap_pto():
    cols = {
        'art': 'varchar',
        'schriftinhalt': 'varchar',
        'dientzurdarstellungvon': 'character(16)[]',
        'wkb_geometry': _GEOM,
    }
    rows = [
        *_changed(_uid('pto', 1), old=dict(wkb_geometry=_point(20, 20)), art='HNR', dientzurdarstellungvon=[_uid('lmh', 1)], wkb_geometry=_point(10, 10)),
        *_changed(_uid('pto', 2), art='HNR', dientzurdarstellungvon=[_uid('lmh', 2)], wkb_geometry=_point(150, 10)),
        *_changed(_uid('pto', 3), art='Strasse', schriftinhalt='Strasse3', dientzurdarstellungvon=[_uid('lmh', 3)], wkb_geometry=_point(250, 10)),
        *_ended(_uid('pto', 4), art='HNR', dientzurdarstellungvon=[_uid('lmh', 4)], wkb_geometry=_point(90, 90)),
    ]
    return cols, rows


def _ax_gebaeude():
    cols = {
        'gebaeudefunktion': 'integer',
        'name': 'varchar[]',
        'grundflaeche': 'double precision',
        'anzahlderoberirdischengeschosse': 'integer',
        'baujahr': 'integer[]',
        'gebaeudekennzeichen': 'varchar',
        'zeigtauf': 'character(16)[]',
        'wkb_geometry': _GEOM,
    }
    rows = [
        *_changed(
            _uid('geb', 1),
            old=dict(grundflaeche=60, anzahlderoberirdischengeschosse=1),
            gebaeudefunktion=1000,
            name=['Gebaeude1'],
            grundflaeche=80,
            anzahlderoberirdischengeschosse=2,
            baujahr=[2000],
            zeigtauf=[_uid('lmh', 1)],
            wkb_geometry=_box(10, 10, 20, 20),
        ),
        *_ended(
            _uid('geb', 2),
            gebaeudefunktion=2000,
            grundflaeche=30,
            zeigtauf=[_uid('lmh', 1)],
            wkb_geometry=_box(30, 30, 35, 36),
        ),
        *_changed(
            _uid('geb', 3),
            old=dict(gebaeudefunktion=1000),
            gebaeudefunktion=1010,
            zeigtauf=[_uid('lmh', 2)],
            wkb_geometry=_box(110, 10, 130, 30),
        ),
    ]
    return cols, rows


##


def _ax_anschrift():
    cols = {
        'strasse': 'varchar',
        'hausnummer': 'varchar',
        'postleitzahlpostzustellung': 'varchar',
        'ort_post': 'varchar',
        'ort_amtlichesortsnamensverzeichnis': 'varchar',
        'ortsteil': 'varchar',
        'bestimmungsland': 'varchar',
        'telefon': 'varchar[]',
    }
    rows = [
        *_changed(
            _uid('ans', 1),
            old=dict(strasse='Strasse5', hausnummer='5'),
            strasse='Strasse4',
            hausnummer='4',
            postleitzahlpostzustellung='00001',
            ort_post='Ort1',
            telefon=['Telefon1'],
        ),
        *_changed(
            _uid('ans', 2),
            old=dict(postleitzahlpostzustellung='00003'),
            strasse='Strasse6',
            hausnummer='6',
            postleitzahlpostzustellung='00002',
            ort_amtlichesortsnamensverzeichnis='Ort2',
            ort_post='Ort2post',
        ),
        *_ended(_uid('ans', 3), strasse='Strasse7', hausnummer='7', postleitzahlpostzustellung='00003', ort_post='Ort3'),
    ]
    return cols, rows


def _ax_person():
    cols = {
        'anrede': 'integer',
        'akademischergrad': 'varchar',
        'nachnameoderfirma': 'varchar',
        'vorname': 'varchar',
        'geburtsname': 'varchar',
        'geburtsdatum': 'date',
        'wohnortodersitz': 'varchar',
        'hat': 'character(16)[]',
    }
    rows = [
        *_changed(
            _uid('pers', 1),
            old=dict(nachnameoderfirma='Geburtsname1', akademischergrad=None),
            anrede=1000,
            akademischergrad='Grad1',
            nachnameoderfirma='Name1',
            vorname='Vorname1',
            geburtsname='Geburtsname1',
            geburtsdatum='1970-01-01',
            hat=[_uid('ans', 1)],
        ),
        *_changed(
            _uid('pers', 2),
            old=dict(nachnameoderfirma='Firma1alt'),
            anrede=3000,
            nachnameoderfirma='Firma1',
            wohnortodersitz='Ort2',
            hat=[_uid('ans', 2)],
        ),
        *_ended(
            _uid('pers', 3),
            anrede=2000,
            nachnameoderfirma='Name3',
            vorname='Vorname3',
            geburtsdatum='1950-01-01',
            hat=[_uid('ans', 3)],
        ),
        *_changed(
            _uid('pers', 4),
            old=dict(vorname='Vorname4alt'),
            anrede=2000,
            nachnameoderfirma='Name4',
            vorname='Vorname4',
        ),
    ]
    return cols, rows


def _ax_namensnummer():
    cols = {
        'laufendenummernachdin1421': 'varchar',
        'nummer': 'varchar',
        'zaehler': 'double precision',
        'nenner': 'double precision',
        'eigentuemerart': 'integer',
        'artderrechtsgemeinschaft': 'integer',
        'beschriebderrechtsgemeinschaft': 'varchar',
        'istbestandteilvon': 'character(16)',
        'benennt': 'character(16)',
    }
    rows = [
        *_changed(
            _uid('nn', 1),
            old=dict(zaehler=1, nenner=1),
            laufendenummernachdin1421='0001',
            zaehler=1,
            nenner=2,
            eigentuemerart=1000,
            istbestandteilvon=_uid('bb', 1),
            benennt=_uid('pers', 1),
        ),
        *_changed(
            _uid('nn', 2),
            old=dict(eigentuemerart=1000),
            laufendenummernachdin1421='0002',
            zaehler=1,
            nenner=2,
            eigentuemerart=1100,
            istbestandteilvon=_uid('bb', 1),
            benennt=_uid('pers', 2),
        ),
        *_ended(
            _uid('nn', 3),
            laufendenummernachdin1421='0003',
            zaehler=1,
            nenner=1,
            eigentuemerart=1000,
            istbestandteilvon=_uid('bb', 1),
            benennt=_uid('pers', 3),
        ),
        *_changed(
            _uid('nn', 4),
            laufendenummernachdin1421='0001',
            eigentuemerart=1000,
            istbestandteilvon=_uid('bb', 2),
            benennt=_uid('pers', 4),
        ),
        *_changed(
            _uid('nn', 5),
            old=dict(beschriebderrechtsgemeinschaft='Rechtsgemeinschaft1alt'),
            laufendenummernachdin1421='0001',
            artderrechtsgemeinschaft=1000,
            beschriebderrechtsgemeinschaft='Rechtsgemeinschaft1',
            istbestandteilvon=_uid('bb', 3),
        ),
        *_changed(
            _uid('nn', 6),
            laufendenummernachdin1421='0001.01',
            zaehler=1,
            nenner=3,
            eigentuemerart=1000,
            istbestandteilvon=_uid('bb', 3),
            benennt=_uid('pers', 1),
        ),
    ]
    return cols, rows


def _ax_buchungsblatt():
    cols = {
        'land': 'varchar',
        'bezirk': 'varchar',
        'buchungsblattkennzeichen': 'varchar',
        'buchungsblattnummermitbuchstabenerweiterung': 'varchar',
        'blattart': 'integer',
    }

    def row(n, blattart=1000):
        return dict(
            land=LAND,
            bezirk=BEZIRK,
            buchungsblattkennzeichen=_buchungsblattkennzeichen(n),
            buchungsblattnummermitbuchstabenerweiterung=f'{n:06d}',
            blattart=blattart,
        )

    rows = [
        *_changed(_uid('bb', 1), old=dict(blattart=2000), **row(1)),
        *_changed(_uid('bb', 2), **row(2)),
        *_changed(_uid('bb', 3), old=dict(blattart=3000), **row(3)),
        *_ended(_uid('bb', 4), **row(4)),
    ]
    return cols, rows


def _ax_buchungsstelle():
    cols = {
        'laufendenummer': 'varchar',
        'buchungsart': 'integer',
        'zaehler': 'double precision',
        'nenner': 'double precision',
        'buchungstext': 'varchar',
        'beschreibungdessondereigentums': 'varchar',
        'beschreibungdesumfangsderbuchung': 'varchar',
        'istbestandteilvon': 'character(16)',
        'an': 'character(16)[]',
        'zu': 'character(16)[]',
    }
    rows = [
        *_changed(
            _uid('bs', 1),
            old=dict(buchungstext='Buchungstext1alt'),
            laufendenummer='0001',
            buchungsart=1100,
            buchungstext='Buchungstext1',
            istbestandteilvon=_uid('bb', 1),
        ),
        *_changed(
            _uid('bs', 2),
            old=dict(buchungsart=2102),
            laufendenummer='0001',
            buchungsart=2101,
            istbestandteilvon=_uid('bb', 2),
            an=[_uid('bs', 1)],
        ),
        *_changed(
            _uid('bs', 3),
            old=dict(zaehler=1, nenner=2),
            laufendenummer='0001',
            buchungsart=1301,
            zaehler=1,
            nenner=4,
            beschreibungdessondereigentums='Sondereigentum1',
            istbestandteilvon=_uid('bb', 3),
            zu=[_uid('bs', 2)],
        ),
        *_changed(
            _uid('bs', 4),
            old=dict(laufendenummer='0010'),
            laufendenummer='0002',
            buchungsart=1100,
            istbestandteilvon=_uid('bb', 1),
        ),
        *_ended(
            _uid('bs', 5),
            laufendenummer='0003',
            buchungsart=1100,
            istbestandteilvon=_uid('bb', 1),
        ),
        *_ended(
            _uid('bs', 6),
            laufendenummer='0001',
            buchungsart=1100,
            istbestandteilvon=_uid('bb', 4),
        ),
    ]
    return cols, rows


##

_FS_COLUMNS = {
    'flurstueckskennzeichen': 'varchar',
    'amtlicheflaeche': 'double precision',
    'flurnummer': 'integer',
    'zaehler': 'varchar',
    'nenner': 'varchar',
    'flurstuecksfolge': 'varchar',
    'land': 'varchar',
    'gemarkungsnummer': 'varchar',
    'gemeindezugehoerigkeit_land': 'varchar',
    'gemeindezugehoerigkeit_regierungsbezirk': 'varchar',
    'gemeindezugehoerigkeit_kreis': 'varchar',
    'gemeindezugehoerigkeit_gemeinde': 'varchar',
    'abweichenderrechtszustand': 'varchar',
    'zweifelhafterflurstuecksnachweis': 'varchar',
    'rechtsbehelfsverfahren': 'varchar',
    'zeitpunktderentstehung': 'date',
    'wkb_geometry': _GEOM,
}


def _fs_row(n, gemarkung, gemeinde, flur, zaehler, nenner=0, folge='', geom=True):
    return dict(
        flurstueckskennzeichen=_flurstueckskennzeichen(gemarkung, flur, zaehler, nenner, folge),
        amtlicheflaeche=10000,
        flurnummer=flur,
        zaehler=str(zaehler),
        nenner=str(nenner) if nenner else None,
        flurstuecksfolge=folge or None,
        land=LAND,
        gemarkungsnummer=gemarkung,
        gemeindezugehoerigkeit_land=LAND,
        gemeindezugehoerigkeit_regierungsbezirk=REGIERUNGSBEZIRK,
        gemeindezugehoerigkeit_kreis=KREIS,
        gemeindezugehoerigkeit_gemeinde=gemeinde,
        zeitpunktderentstehung='2000-01-01',
        wkb_geometry=_fs_box(n) if geom else None,
    )


def _ax_flurstueck():
    cols = _FS_COLUMNS | {
        'zustaendigestelle_land': 'varchar[]',
        'zustaendigestelle_stelle': 'varchar[]',
        'istgebucht': 'character(16)',
        'zeigtauf': 'character(16)[]',
        'weistauf': 'character(16)[]',
    }
    rows = [
        *_changed(
            _uid('fs', 1),
            old=dict(amtlicheflaeche=9990, istgebucht=_uid('bs', 5), weistauf=[_uid('lmh', 4)]),
            **_fs_row(1, GEMARKUNG_1, GEMEINDE_1, 1, 1),
            zustaendigestelle_land=[LAND],
            zustaendigestelle_stelle=[STELLE],
            istgebucht=_uid('bs', 1),
            weistauf=[_uid('lmh', 1), _uid('lmh', 4)],
            zeigtauf=[_uid('loh', 1)],
        ),
        *_changed(
            _uid('fs', 2),
            old=dict(abweichenderrechtszustand='true'),
            **_fs_row(2, GEMARKUNG_1, GEMEINDE_1, 1, 2, nenner=1),
            istgebucht=_uid('bs', 3),
            weistauf=[_uid('lmh', 2), _uid('lmh', 5), _uid('lmh', 6), _uid('lmh', 7), _uid('lmh', 8)],
        ),
        *_changed(
            _uid('fs', 3),
            old=dict(gemeindezugehoerigkeit_gemeinde=GEMEINDE_1),
            **_fs_row(3, GEMARKUNG_2, GEMEINDE_2, 2, 3, nenner=2, folge='01'),
            istgebucht=_uid('bs', 4),
            weistauf=[_uid('lmh', 3)],
            zeigtauf=[_uid('loh', 2)],
        ),
        *_ended(
            _uid('fs', 4),
            **_fs_row(4, GEMARKUNG_1, GEMEINDE_1, 1, 4),
            istgebucht=_uid('bs', 6),
        ),
        *_changed(
            _uid('fs', 5),
            **_fs_row(5, GEMARKUNG_1, GEMEINDE_3, 1, 5),
        ),
        *_changed(
            _uid('fs', 6),
            **_fs_row(6, GEMARKUNG_1, GEMEINDE_1, 1, 6, geom=False),
        ),
        *_changed(
            _uid('fs', 7),
            old=dict(amtlicheflaeche=10010),
            **_fs_row(7, GEMARKUNG_3, GEMEINDE_1, 3, 7),
        ),
    ]
    return cols, rows


def _ax_historischesflurstueck():
    cols = _FS_COLUMNS | {
        'nachfolgerflurstueckskennzeichen': 'varchar[]',
        'zeitpunktderhistorisierung': 'date',
        'blattart': 'integer[]',
        'buchungsart': 'varchar[]',
        'buchungsblattbezirk_land': 'varchar[]',
        'buchungsblattbezirk_bezirk': 'varchar[]',
        'buchungsblattkennzeichen': 'varchar[]',
        'buchungsblattnummermitbuchstabenerweiterung': 'varchar[]',
        'laufendenummerderbuchungsstelle': 'varchar[]',
    }
    rows = [
        dict(
            gml_id=_uid('hfs', 1),
            beginnt=BEGIN_OLD,
            anlass=[ANLASS_OLD],
            **_fs_row(8, GEMARKUNG_1, GEMEINDE_1, 1, 8),
            zeitpunktderhistorisierung='2025-06-01',
            nachfolgerflurstueckskennzeichen=[
                _flurstueckskennzeichen(GEMARKUNG_1, 1, 1),
                _flurstueckskennzeichen(GEMARKUNG_1, 1, 2, nenner=1),
                _flurstueckskennzeichen(GEMARKUNG_1, 1, 99),
            ],
            blattart=[1000, 1000],
            buchungsart=['1100', '1100'],
            buchungsblattbezirk_land=[LAND, LAND],
            buchungsblattbezirk_bezirk=[BEZIRK, BEZIRK],
            buchungsblattkennzeichen=[_buchungsblattkennzeichen(4), _buchungsblattkennzeichen(1)],
            buchungsblattnummermitbuchstabenerweiterung=[f'{4:06d}', f'{1:06d}'],
            laufendenummerderbuchungsstelle=['0001', '0009'],
        ),
        dict(
            gml_id=_uid('hfs', 2),
            beginnt=BEGIN_OLD,
            anlass=[ANLASS_OLD],
            **_fs_row(9, GEMARKUNG_1, GEMEINDE_1, 1, 9),
            zeitpunktderhistorisierung='2025-06-01',
            nachfolgerflurstueckskennzeichen=[_flurstueckskennzeichen(GEMARKUNG_1, 1, 1)],
        ),
    ]
    return cols, rows


##

_PART_COLUMNS = {
    'wkb_geometry': _GEOM,
}


def _ax_wohnbauflaeche():
    cols = _PART_COLUMNS | {
        'artderbebauung': 'integer',
        'zustand': 'integer',
        'name': 'varchar',
    }
    rows = [
        *_changed(
            _uid('wohn', 1),
            old=dict(wkb_geometry=_box(0, 0, 100, 100), artderbebauung=2000),
            artderbebauung=1000,
            name='Wohnbauflaeche1',
            wkb_geometry=_box(0, 0, 200, 100),
        ),
    ]
    return cols, rows


def _ax_strassenverkehr():
    cols = _PART_COLUMNS | _LAGE_KEY_COLUMNS | {
        'funktion': 'integer',
        'unverschluesselt': 'varchar',
        'zweitname': 'varchar',
    }
    rows = [
        *_changed(
            _uid('str', 1),
            old=dict(funktion=2312),
            funktion=2311,
            unverschluesselt='Strasse3',
            wkb_geometry=_box(200, 0, 300, 100),
        ),
        *_ended(
            _uid('str', 2),
            funktion=2311,
            wkb_geometry=_box(200, 50, 300, 100),
            **_lage_key(GEMEINDE_1, LAGE_1),
        ),
    ]
    return cols, rows


def _ax_bodenschaetzung():
    cols = _PART_COLUMNS | {
        'ackerzahlodergruenlandzahl': 'varchar',
        'bodenzahlodergruenlandgrundzahl': 'varchar',
        'bodenart': 'integer',
        'nutzungsart': 'integer',
        'jahreszahl': 'integer',
    }
    rows = [
        *_changed(
            _uid('bod', 1),
            old=dict(ackerzahlodergruenlandzahl='40'),
            ackerzahlodergruenlandzahl='45',
            bodenzahlodergruenlandgrundzahl='50',
            bodenart=2100,
            nutzungsart=1000,
            jahreszahl=1950,
            wkb_geometry=_box(200, 0, 250, 100),
        ),
    ]
    return cols, rows


def _ax_denkmalschutzrecht():
    cols = _PART_COLUMNS | {
        'artderfestlegung': 'integer',
        'land': 'varchar',
        'stelle': 'varchar',
        'bezeichnung': 'varchar',
        'name': 'varchar',
    }
    rows = [
        *_changed(
            _uid('denk', 1),
            old=dict(artderfestlegung=2700),
            artderfestlegung=2711,
            name='Denkmal1',
            land=LAND,
            stelle=STELLE,
            wkb_geometry=_box(50, 0, 100.005, 50),
        ),
    ]
    return cols, rows


if __name__ == '__main__':
    generate(sys.argv[1])
