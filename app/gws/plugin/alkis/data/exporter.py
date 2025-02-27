"""ALKIS exporter.

Export Flurstuecke to CSV or GeoJSON.
"""

from typing import Optional, Iterable, cast, Any

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


class GroupConfig(gws.Config):
    """Export group configuration."""

    title: str
    """Title to display in the ui"""
    fields: list[gws.ext.config.modelField]
    """Fields to export."""


class Config(gws.ConfigWithAccess):
    """Export configuration"""

    groups: Optional[list[GroupConfig]]
    """Export groups"""


class Group(gws.Data):
    index: int
    title: str
    withEigentuemer: bool
    withBuchung: bool
    fieldNames: list[str]
    keys: set[str]


class Format(gws.Enum):
    csv = 'csv'
    geojson = 'geojson'
    pydict = 'pydict'


class Args(gws.Data):
    format: Format
    fs: Iterable[dt.Flurstueck]
    groups: Optional[list[Group]]
    writePath: Optional[str]
    user: gws.User
    progress: Optional[ProgressIndicator]


_DEFAULT_GROUPS = [
    gws.Config(
        title='Basisdaten',
        fields=[
            gws.Config(type='text', name='fs_flurstueckskennzeichen', title='Flurstückskennzeichen'),
            gws.Config(type='text', name='fs_recs_gemeinde_text', title='Gemeinde'),
            gws.Config(type='text', name='fs_recs_gemarkung_code', title='Gemarkungsnummer'),
            gws.Config(type='text', name='fs_recs_gemarkung_text', title='Gemarkung'),
            gws.Config(type='text', name='fs_recs_flurnummer', title='Flurnummer'),
            gws.Config(type='text', name='fs_recs_zaehler', title='Zähler'),
            gws.Config(type='text', name='fs_recs_nenner', title='Nenner'),
            gws.Config(type='text', name='fs_recs_flurstuecksfolge', title='Folge'),
            gws.Config(type='float', name='fs_recs_amtlicheFlaeche', title='Fläche'),
            gws.Config(type='float', name='fs_recs_x', title='X'),
            gws.Config(type='float', name='fs_recs_y', title='Y'),
        ]
    ),
    gws.Config(
        title='Lage',
        fields=[
            gws.Config(type='text', name='fs_lageList_recs_strasse', title='FS Strasse'),
            gws.Config(type='text', name='fs_lageList_recs_hausnummer', title='FS Hnr'),
        ]
    ),
    gws.Config(
        title='Gebäude',
        fields=[
            gws.Config(type='float', name='fs_gebaeudeList_recs_geomFlaeche', title='Gebäude Fläche'),
            gws.Config(type='text', name='fs_gebaeudeList_recs_props_gebaeudefunktion_text', title='Gebäude Funktion'),
        ]
    ),
    gws.Config(
        title='Buchungsblatt',
        fields=[
            gws.Config(type='text', name='fs_buchungList_recs_buchungsstelle_recs_buchungsart_code', title='Buchungsart'),
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_recs_blattart_text', title='Blattart'),
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_recs_buchungsblattkennzeichen', title='Blattkennzeichen'),
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_recs_buchungsblattnummerMitBuchstabenerweiterung', title='Blattnummer'),
            gws.Config(type='text', name='fs_buchungList_recs_buchungsstelle_laufendeNummer', title='Laufende Nummer'),
        ]
    ),
    gws.Config(
        title='Eigentümer',
        fields=[
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_namensnummerList_recs_anteil', title='Anteil'),
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_namensnummerList_personList_recs_vorname', title='Vorname'),
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_namensnummerList_personList_recs_nachnameOderFirma', title='Name'),
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_namensnummerList_personList_recs_geburtsdatum', title='Geburtsdatum'),
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_namensnummerList_personList_anschriftList_recs_strasse', title='Strasse'),
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_namensnummerList_personList_anschriftList_recs_hausnummer', title='Hnr'),
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_namensnummerList_personList_anschriftList_recs_plz', title='PLZ'),
            gws.Config(type='text', name='fs_buchungList_buchungsblatt_namensnummerList_personList_anschriftList_recs_ort', title='Ort'),
        ]
    ),
    gws.Config(
        title='Nutzung',
        fields=[
            gws.Config(type='float', name='fs_nutzungList_geomFlaeche', title='Nutzung Fläche'),
            gws.Config(type='text', name='fs_nutzungList_name_text', title='Nutzung Typ'),
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

        for cfg in (self.cfg('groups') or _DEFAULT_GROUPS):
            g = Group(
                index=len(self.groups),
                title=cfg.title,
                fieldNames=[],
                keys=set(),
                withEigentuemer=False,
                withBuchung=False
            )
            self.groups.append(g)

            for fc in cfg.fields:
                field_configs[fc.name] = fc
                g.fieldNames.append(fc.name)
                self._update_keys_from_field(fc, g.keys, flat_keys)

            g.withEigentuemer = any('namensnummer' in k for k in g.keys)
            g.withBuchung = any('buchung' in k for k in g.keys)

        self.model = self.create_child(
            Model,
            fields=list(field_configs.values())
        )

    def _update_keys_from_field(self, fc, keys, flat_keys):
        # extract export keys `fs_...` from a field config

        # field name is a valid key - ok
        if fc.name in flat_keys:
            keys.add(fc.name)

        # a pretty crude way to extract keys from dynamic values like `format` or `expression`
        for val in (fc.values or []):
            ref = val.format or val.text
            if ref:
                for k in flat_keys:
                    if k in ref:
                        keys.add(k)

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

        header = []

        for row in self._iter_rows(args):
            if not header:
                header = list(row.keys())
                writer.write_headers(header)
            writer.write_row([row.get(k, '') for k in header])

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

        all_keys = set()
        field_names = []

        for g in args.groups:
            all_keys.update(g.keys)
            field_names.extend(g.fieldNames)

        fields = [
            fld
            for fn in gws.u.uniq(field_names)
            for fld in self.model.fields
            if fld.name == fn
        ]

        mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.searchResults, user=args.user)
        row_hashes = set()

        for fs in args.fs:
            if args.progress:
                args.progress.update(1)

            for atts in _flatten(fs, all_keys):
                # create a 'raw' feature from attributes and convert it to a record
                # so that dynamic fields can be computed

                feature = gws.base.feature.new(model=self.model, attributes=atts)
                for fld in fields:
                    fld.to_record(feature, mc)

                row = {
                    fld.title: feature.record.attributes.get(fld.name, '')
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


def _flatten(fs, all_keys) -> list[dict[str, Any]]:
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
