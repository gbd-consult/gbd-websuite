from typing import Optional, cast

import gws
import gws.base.model
import gws.base.feature
import gws.helper.csv
import gws.lib.intl

from . import types as dt


class Config(gws.ConfigWithAccess):
    """CSV export configuration"""

    models: Optional[list[gws.ext.config.model]]
    """export groups"""


_DEFAULT_MODELS = [
    gws.Config(
        title='Basisdaten',
        fields=[
            gws.Config(type='text', name='fs_recs_gemeinde_text', title='Gemeinde'),
            gws.Config(type='text', name='fs_recs_gemarkung_code', title='Gemarkungsnummer'),
            gws.Config(type='text', name='fs_recs_gemarkung_text', title='Gemarkung'),
            gws.Config(type='text', name='fs_recs_flurnummer', title='Flurnummer'),
            gws.Config(type='text', name='fs_recs_zaehler', title='Zähler'),
            gws.Config(type='text', name='fs_recs_nenner', title='Nenner'),
            gws.Config(type='text', name='fs_recs_flurstuecksfolge', title='Folge'),
            gws.Config(type='text', name='fs_recs_amtlicheFlaeche', title='Fläche'),
            gws.Config(type='text', name='fs_recs_x', title='X'),
            gws.Config(type='text', name='fs_recs_y', title='Y'),
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
            gws.Config(type='text', name='fs_gebaeudeList_recs_area', title='Gebäude Fläche'),
            gws.Config(type='text', name='fs_gebaeudeList_recs_props_Gebäudefunktion_text', title='Gebäude Funktion'),
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
            gws.Config(type='text', name='fs_nutzungList_area', title='Nutzung Fläche'),
            gws.Config(type='text', name='fs_nutzungList_name_text', title='Nutzung Typ'),
        ]
    ),
]


class Group(gws.Data):
    index: int
    title: str
    withEigentuemer: bool
    withBuchung: bool
    fieldNames: list[str]


class Model(gws.base.model.Object):
    def configure(self):
        self.configure_model()


class Object(gws.Node):
    model: Model
    groups: list[Group]

    def configure(self):
        self.groups = []

        fields_map = {}

        p = self.cfg('models') or _DEFAULT_MODELS
        for n, cfg in enumerate(p, 1):
            self.groups.append(Group(
                index=n,
                title=cfg.title,
                fieldNames=[f.name for f in cfg.fields],
                withEigentuemer=any('namensnummer' in f.name for f in cfg.fields),
                withBuchung=any('buchung' in f.name for f in cfg.fields),
            ))
            for f in cfg.fields:
                if f.name not in fields_map:
                    fields_map[f.name] = f

        self.model = self.create_child(
            Model,
            fields=list(fields_map.values())
        )

    """
    The Flurstueck structure, as created by our indexer, is deeply nested.
    We flatten it first, creating a dicts 'nested_key->value'. For list values, we repeat the dict 
    for each item in the list, thus creating a product of all lists, e.g.
    
        record:  
            a:x, b:[1,2], c:[3,4]
    
        flat list:
            a:x, b:1, c:3
            a:x, b:1, c:4
            a:x, b:2, c:3
            a:x, b:2, c:4
        
    Then we apply our composite model to each element in the flat list.
    
    Finally, keep only keys which are members in the requested models.
    """

    def export_as_csv(self, fs_list: list[dt.Flurstueck], groups: list[Group], user: gws.User):

        field_names = []

        for g in groups:
            for s in g.fieldNames:
                if s not in field_names:
                    field_names.append(s)

        fields = [
            fld
            for name in field_names
            for fld in self.model.fields
            if fld.name == name
        ]

        csv_helper = cast(gws.helper.csv.Object, self.root.app.helper('csv'))
        writer = csv_helper.writer(gws.lib.intl.locale('de_DE'))

        writer.write_headers([fld.title for fld in fields])
        mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.searchResults, user=user)

        for n, fs in enumerate(fs_list, 1):
            gws.log.debug(f'export {n}/{len(fs_list)} {fs.uid=}')
            row_hashes = set()
            for atts in _flatten(fs):
                rec = gws.FeatureRecord(attributes=atts)
                feature = gws.base.feature.new(model=self.model, record=rec)
                for fld in fields:
                    fld.from_record(feature, mc)
                row = [feature.get(fld.name, '') for fld in fields]
                h = gws.u.sha256(row)
                if h not in row_hashes:
                    row_hashes.add(h)
                    writer.write_row(row)

        return writer.to_bytes()


def _flatten(obj):
    def flat(o, key, ds):
        if isinstance(o, list):
            if not o:
                return ds
            ds2 = []
            for v in o:
                for d2 in flat(v, key, [{}]):
                    for d in ds:
                        ds2.append(d | d2)
            return ds2

        if isinstance(o, (dt.Object, dt.EnumPair)):
            for k, v in vars(o).items():
                if k == 'fsUids':
                    # exclude, it's basically the same as flurstueckskennzeichenList
                    continue
                if k == 'props':
                    # the 'props' element, which is a list of key-value pairs
                    # requires a special treatment
                    for k2, v2 in v:
                        ds = flat(v2, f'{key}_props_{k2}', ds)
                else:
                    ds = flat(v, f'{key}_{k}', ds)
            return ds

        for d in ds:
            d[key] = o
        return ds

    return flat(obj, 'fs', [{}])
