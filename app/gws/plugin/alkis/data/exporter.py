"""ALKIS exporter.

Export Flurstuecke to CSV or GeoJSON.
"""

from typing import Optional, Iterable, cast

import io

import gws
import gws.base.model
import gws.base.feature
import gws.plugin.csv_helper
import gws.lib.crs
import gws.lib.intl
import gws.lib.jsonx

from gws.lib.cli import ProgressIndicator
from . import types as dt


class FieldConfig(gws.Config):
    """Export field configuration."""

    title: str
    """Field title."""
    key: str
    """Flattened key name."""


class GroupConfig(gws.Config):
    """Export group configuration."""

    title: str
    """Title to display in the ui"""
    fields: list[FieldConfig]
    """Fields to export."""


class Group(gws.Data):
    index: int
    title: str
    withEigentuemer: bool
    withBuchung: bool
    keys: list[str]


class Format(gws.Enum):
    csv = 'csv'
    geojson = 'geojson'
    pydict = 'pydict'


class Args(gws.Data):
    """Export arguments."""

    format: Format
    fs: Iterable[dt.Flurstueck]
    groups: Optional[list[Group]]
    writePath: Optional[str]
    user: gws.User
    progress: Optional[ProgressIndicator]


class Config(gws.ConfigWithAccess):
    """Export configuration"""

    groups: Optional[list[GroupConfig]]
    """Export groups"""


_DEFAULT_GROUPS = [
    gws.Config(
        title='Basisdaten',
        fields=[
            gws.Config(key='fs_flurstueckskennzeichen', title='Flurstückskennzeichen'),
            gws.Config(key='fs_recs_gemeinde_text', title='Gemeinde'),
            gws.Config(key='fs_recs_gemarkung_code', title='Gemarkungsnummer'),
            gws.Config(key='fs_recs_gemarkung_text', title='Gemarkung'),
            gws.Config(key='fs_recs_flurnummer', title='Flurnummer'),
            gws.Config(key='fs_recs_zaehler', title='Zähler'),
            gws.Config(key='fs_recs_nenner', title='Nenner'),
            gws.Config(key='fs_recs_flurstuecksfolge', title='Folge'),
            gws.Config(key='fs_recs_amtlicheFlaeche', title='Fläche'),
            gws.Config(key='fs_recs_x', title='X'),
            gws.Config(key='fs_recs_y', title='Y'),
        ]
    ),
    gws.Config(
        title='Lage',
        fields=[
            gws.Config(key='fs_lageList_recs_strasse', title='FS Strasse'),
            gws.Config(key='fs_lageList_recs_hausnummer', title='FS Hnr'),
        ]
    ),
    gws.Config(
        title='Gebäude',
        fields=[
            gws.Config(key='fs_gebaeudeList_recs_geomFlaeche', title='Gebäude Fläche'),
            gws.Config(key='fs_gebaeudeList_recs_props_gebaeudefunktion_text', title='Gebäude Funktion'),
        ]
    ),
    gws.Config(
        title='Buchungsblatt',
        fields=[
            gws.Config(key='fs_buchungList_recs_buchungsstelle_recs_buchungsart_code', title='Buchungsart'),
            gws.Config(key='fs_buchungList_buchungsblatt_recs_blattart_text', title='Blattart'),
            gws.Config(key='fs_buchungList_buchungsblatt_recs_buchungsblattkennzeichen', title='Blattkennzeichen'),
            gws.Config(key='fs_buchungList_buchungsblatt_recs_buchungsblattnummerMitBuchstabenerweiterung', title='Blattnummer'),
            gws.Config(key='fs_buchungList_recs_buchungsstelle_laufendeNummer', title='Laufende Nummer'),
        ]
    ),
    gws.Config(
        title='Eigentümer',
        fields=[
            gws.Config(key='fs_buchungList_buchungsblatt_namensnummerList_recs_anteil', title='Anteil'),
            gws.Config(key='fs_buchungList_buchungsblatt_namensnummerList_personList_recs_vorname', title='Vorname'),
            gws.Config(key='fs_buchungList_buchungsblatt_namensnummerList_personList_recs_nachnameOderFirma', title='Name'),
            gws.Config(key='fs_buchungList_buchungsblatt_namensnummerList_personList_recs_geburtsdatum', title='Geburtsdatum'),
            gws.Config(key='fs_buchungList_buchungsblatt_namensnummerList_personList_anschriftList_recs_strasse', title='Strasse'),
            gws.Config(key='fs_buchungList_buchungsblatt_namensnummerList_personList_anschriftList_recs_hausnummer', title='Hnr'),
            gws.Config(key='fs_buchungList_buchungsblatt_namensnummerList_personList_anschriftList_recs_plz', title='PLZ'),
            gws.Config(key='fs_buchungList_buchungsblatt_namensnummerList_personList_anschriftList_recs_ort', title='Ort'),
        ]
    ),
    gws.Config(
        title='Nutzung',
        fields=[
            gws.Config(key='fs_nutzungList_geomFlaeche', title='Nutzung Fläche'),
            gws.Config(key='fs_nutzungList_name_text', title='Nutzung Typ'),
        ]
    ),
]


class Model(gws.base.model.Object):
    def configure(self):
        self.configure_model()


class Object(gws.Node):
    model: Model
    groups: list[Group]

    def configure(self):
        self.groups = []

        flat_keys = get_flat_keys()
        field_configs = {}

        _cls_to_type = {
            int: 'integer',
            float: 'float',
            str: 'text',
        }

        p = self.cfg('groups') or _DEFAULT_GROUPS

        for n, cfg in enumerate(p, 1):
            keys = []

            for f in cfg.fields:
                cls = flat_keys.get(f.key)
                if not cls:
                    raise gws.ConfigurationError(f'invalid ALKIS export field {f.key!r}')

                field_configs[f.key] = gws.Config(
                    type=_cls_to_type.get(cls, 'text'),
                    title=f.title,
                    name=f.key,
                )

                keys.append(f.key)

            self.groups.append(Group(
                index=n,
                title=cfg.title,
                keys=keys,
                withEigentuemer=any('namensnummer' in f for f in keys),
                withBuchung=any('buchung' in f for f in keys),
            ))

        self.model = self.create_child(
            Model,
            fields=list(field_configs.values())
        )

    def run(self, args: Args):
        """Export a Flurstueck list to CSV or GeoJSON."""

        if args.format == Format.csv:
            return self._export_csv(args)
        if args.format == Format.geojson:
            return self._export_geojson(args)
        if args.format == Format.pydict:
            return self._export_dict(args)

    def _export_csv(self, args: Args):
        csv_helper = cast(gws.plugin.csv_helper.Object, self.root.app.helper('csv'))

        fp = open(args.writePath, 'wb') if args.writePath else None
        writer = csv_helper.writer(gws.lib.intl.locale('de_DE'), stream_to=fp)

        keys = []

        for row in self._iter_rows(args):
            if not keys:
                keys = list(row.keys())
                writer.write_headers(keys)
            writer.write_row([row.get(k, '') for k in keys])

        if args.writePath:
            fp.close()
            return

        return writer.to_bytes()

    def _export_geojson(self, args: Args):
        fp = open(args.writePath, 'wb') if args.writePath else io.BytesIO()
        fp.write(b'{"type": "FeatureCollection", "features": [')

        comma = b'\n '

        for row in self._iter_rows(args, with_geometry=True):
            d = self._row_to_dict(row)
            fp.write(comma + gws.lib.jsonx.to_string(d).encode('utf8'))
            comma = b',\n '

        fp.write(b'\n]}\n')

        if args.writePath:
            fp.close()
            return

        b = fp.getvalue()
        fp.close()
        return b

    def _export_dict(self, args: Args):
        return dict(
            type='FeatureCollection',
            features=[
                self._row_to_dict(row)
                for row in self._iter_rows(args, with_geometry=True)
            ]
        )

    def _row_to_dict(self, row):
        shape = row.pop('fs_shape', None)
        return dict(
            type='Feature',
            properties=row,
            geometry=cast(gws.Shape, shape).to_geojson() if shape else None,
        )

    def _iter_rows(self, args: Args, with_geometry=False):
        """Iterate over a Flurstueck list and yield flat rows (dicts).

        The Flurstueck structure, as created by our indexer, is deeply nested.
        We flatten it, creating a dict 'nested_key->value'. For list values, we repeat the dict
        for each item in the list, thus creating a product of all lists, e.g.

        record:
            a:x, b:[1,2], c:[3,4]

        flat list:
            a:x, b:1, c:3
            a:x, b:1, c:4
            a:x, b:2, c:3
            a:x, b:2, c:4

        @TODO: with certain combinations of keys this can explode very quickly
        """

        all_keys = []

        for g in args.groups:
            for k in g.keys:
                if k not in all_keys:
                    all_keys.append(k)

        fields = [
            fld
            for key in all_keys
            for fld in self.model.fields
            if fld.name == key
        ]

        mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.searchResults, user=args.user)
        row_hashes = set()

        for fs in args.fs:
            if args.progress:
                args.progress.update(1)

            for atts in _flatten(fs, all_keys):
                rec = gws.FeatureRecord(attributes=atts)
                feature = gws.base.feature.new(model=self.model, record=rec)

                for fld in fields:
                    fld.from_record(feature, mc)

                row = {
                    fld.title: feature.get(fld.name, '')
                    for fld in fields
                }

                h = gws.u.sha256(row)
                if h in row_hashes:
                    continue

                row_hashes.add(h)
                if with_geometry:
                    row['fs_shape'] = fs.shape
                yield row


_EXCLUDE_KEYS = {'fsUids', 'childUids', 'parentUids'}


def _flatten(fs, all_keys):
    def _flat(val, key, ds):

        if not any(k.startswith(key) for k in all_keys):
            return ds

        if isinstance(val, list):
            if not val:
                return ds

            ds2 = []
            for v in val:
                for d2 in _flat(v, key, [{}]):
                    for d in ds:
                        ds2.append(d | d2)
            return ds2

        if isinstance(val, (dt.Object, dt.EnumPair)):
            for k, v in vars(val).items():
                if k not in _EXCLUDE_KEYS:
                    ds = _flat(v, f'{key}_{k}', ds)
            return ds

        for d in ds:
            d[key] = val

        return ds

    return _flat(fs, 'fs', [{}])


def get_flat_keys():
    """Return a dict key->type for all flat keys in the Flurstueck structure."""

    return {
        k: typ
        for k, typ in sorted(set(_get_keys(dt.Flurstueck, 'fs')))
    }


def _get_keys(cls, key):
    if isinstance(cls, str):
        cls = getattr(dt, cls, None)

    if cls is dt.EnumPair:
        yield f'{key}_code', int
        yield f'{key}_text', str
        return

    if not cls or not hasattr(cls, '__annotations__'):
        yield key, cls or str
        return

    for k, typ in cls.__annotations__.items():
        if k in _EXCLUDE_KEYS:
            continue
        if getattr(typ, '__origin__', None) is list:
            yield from _get_keys(typ.__args__[0], f'{key}_{k}')
        else:
            yield from _get_keys(typ, f'{key}_{k}')
