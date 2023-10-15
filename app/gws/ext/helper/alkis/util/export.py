import itertools
import gws.ext.helper.csv
import gws.common.model

import gws
import gws.types as t

"""
A fs structure, as created by our indexer, is deeply nested.
We flatten it first, creating a list 'some.nested.key, list positions, value'
    
    ...then filter out unwanted keys
    
    ...then create a product of all list positions, so if there are 3 'lage' lists
    and 2 'eigentuemer' lists, there will be 3x2=6 rows
"""


class GroupConfig(t.Config):
    """Export group configuration"""

    title: str  #: title for this group
    eigentuemer: bool = False  #: include Eigentuemer (owner) data
    buchung: bool = False  #: include Grundbuch (register) data
    model: t.Optional[gws.common.model.Config]  #: data model for this group


class Config(t.WithAccess):
    """CSV Export configuration"""

    groups: t.Optional[t.List[GroupConfig]]  #: export groups


# default export groups configuration


_Model = gws.common.model.Config
_Field = gws.common.model.FieldConfig

DEFAULT_GROUPS = [
    t.Config(
        title='Basisdaten', eigentuemer=False, buchung=False,
        model=_Model(fields=[
            _Field(name='gemeinde', title='Gemeinde', type='string'),
            _Field(name='gemarkung_id', title='Gemarkung ID', type='string'),
            _Field(name='gemarkung', title='Gemarkung', type='string'),
            _Field(name='flurnummer', title='Flurnummer', type='integer'),
            _Field(name='zaehler', title='Zähler', type='integer'),
            _Field(name='nenner', title='Nenner', type='string'),
            _Field(name='flurstuecksfolge', title='Folge', type='string'),
            _Field(name='amtlicheflaeche', title='Fläche', type='float'),
            _Field(name='x', title='X', type='float'),
            _Field(name='y', title='Y', type='float'),
        ])
    ),
    t.Config(
        title='Lage', eigentuemer=False, buchung=False,
        model=_Model(fields=[
            _Field(name='lage_strasse', title='FS Strasse', type='string'),
            _Field(name='lage_hausnummer', title='FS Hnr', type='string'),
        ])
    ),
    t.Config(
        title='Gebäude', eigentuemer=False, buchung=False,
        model=_Model(fields=[
            _Field(name='gebaeude_area', title='Gebäude Fläche', type='float'),
            _Field(name='gebaeude_gebaeudefunktion', title='Gebäude Funktion', type='string'),
        ])
    ),
    t.Config(
        title='Buchungsblatt', eigentuemer=False, buchung=True,
        model=_Model(fields=[
            _Field(name='buchung_buchungsart', title='Buchungsart', type='string'),
            _Field(name='buchung_buchungsblatt_blattart', title='Blattart', type='string'),
            _Field(name='buchung_buchungsblatt_buchungsblattkennzeichen', title='Blattkennzeichen', type='string'),
            _Field(name='buchung_buchungsblatt_buchungsblattnummermitbuchstabenerweiterung', title='Blattnummer', type='string'),
            _Field(name='buchung_laufendenummer', title='Laufende Nummer', type='string'),
        ])
    ),
    t.Config(
        title='Eigentümer', eigentuemer=True, buchung=True,
        model=_Model(fields=[
            _Field(name='buchung_eigentuemer_person_vorname', title='Vorname', type='string'),
            _Field(name='buchung_eigentuemer_person_nachnameoderfirma', title='Name', type='string'),
            _Field(name='buchung_eigentuemer_person_geburtsdatum', title='Geburtsdatum', type='string'),
            _Field(name='buchung_eigentuemer_person_anschrift_strasse', title='Strasse', type='string'),
            _Field(name='buchung_eigentuemer_person_anschrift_hausnummer', title='Hnr', type='string'),
            _Field(name='buchung_eigentuemer_person_anschrift_postleitzahlpostzustellung', title='PLZ', type='string'),
            _Field(name='buchung_eigentuemer_person_anschrift_ort_post', title='Ort', type='string'),
        ])
    ),
    t.Config(
        title='Nutzung', eigentuemer=False, buchung=False,
        model=_Model(fields=[
            _Field(name='nutzung_a_area', title='Nutzung Fläche', type='float'),
            _Field(name='nutzung_type', title='Nutzung Typ', type='string'),
        ])
    ),
]


class Group(t.Data):
    index: int
    title: str
    withEigentuemer: bool
    withBuchung: bool
    fieldNames: set


class Object(gws.Object):
    model: gws.common.model.Object
    groups: t.List[Group]

    def configure(self):
        ls = self.var('groups') or DEFAULT_GROUPS
        all_fields = {}

        self.groups = []

        for n, cfg in enumerate(ls, 1):
            group = Group(
                index=n,
                title=cfg.title,
                withEigentuemer=cfg.eigentuemer is True,
                withBuchung=cfg.buchung is True,
                fieldNames=set()
            )
            for f in cfg.model.fields:
                if f.name not in all_fields:
                    all_fields[f.name] = f
                group.fieldNames.add(f.name)

            self.groups.append(group)

        self.model = t.cast(
            gws.common.model.Object,
            self.root.create_child(gws.common.model.Object, t.Config(fields=list(all_fields.values()))))

    def group_dict(self, with_eigentuemer: bool, with_buchung: bool):
        d = {}

        for group in self.groups:
            if group.withEigentuemer and not with_eigentuemer:
                continue
            if group.withBuchung and not with_buchung:
                continue
            d[group.index] = group.title

        return d

    def export_as_csv(self, fs_features: t.List[t.IFeature], group_indexes: t.List[int]):

        fset = set()

        for group in self.groups:
            if group.index in group_indexes:
                fset.update(group.fieldNames)

        fields = [f for f in self.model.fields if f.name in fset]
        field_names = [f.name for f in fields]

        helper: gws.ext.helper.csv.Object = t.cast(
            gws.ext.helper.csv.Object,
            self.root.find_first('gws.ext.helper.csv'))
        writer = helper.writer()

        writer.write_headers([f.title for f in fields])

        mc = t.ModelContext()
        for fs in fs_features:
            for rec in _recs_from_feature(fs, field_names):
                f = self.model.feature_from_props(t.FeatureProps(attributes=rec), mc)
                writer.write_dict(field_names, f.attributes)

        return writer.as_bytes()


def _recs_from_feature(fs: t.IFeature, att_names: t.List[str]):
    # create a flat list from the attributes of the FS feature

    flat = list(_flat_walk(fs.attributes))

    # keep keys we need

    flat = [e for e in flat if any(e['path'].startswith(a) for a in att_names)]

    # compute max index for each list from 'pos' elements

    max_index = {}

    for e in flat:
        for k, v in e['pos'].items():
            max_index[k] = max(max_index.get(k, 0), v)

    # no lists, return a single record

    if not max_index:
        yield {e['path']: e['value'] for e in flat}
        return

    # create a record for each combination of list indexes

    list_keys = max_index.keys()
    list_ranges = [range(x + 1) for x in max_index.values()]

    for list_indexes in itertools.product(*list_ranges):
        matching = [e for e in flat if _indexes_match(e, list_keys, list_indexes)]
        yield {e['path']: e['value'] for e in matching}


def _flat_walk(obj, path=None, pos=None):
    # create a flat list from a nested fs record
    # an element of the list is {path, pos, value}, where
    #     path = full key path (joined by _)
    #     pos  = {list_name: list_index, ...} if a value is a member of a list
    #     value = element value

    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat_walk(v, path + '_' + str(k) if path else str(k), pos)
        return

    if isinstance(obj, list):
        for n, v in enumerate(obj):
            p = dict(pos) if pos else {}
            p[path] = n
            yield from _flat_walk(v, path, p)
        return

    yield {'path': path, 'pos': pos or {}, 'value': obj}


def _indexes_match(flat_elem, list_keys, list_indexes):
    # check if a flat entry matches the given combination of list positions

    for k, i in zip(list_keys, list_indexes):
        p = flat_elem['pos'].get(k)
        if p is not None and p != i:
            return False

    return True
