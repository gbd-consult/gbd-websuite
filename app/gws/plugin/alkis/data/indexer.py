import re
from typing import Generic, TypeVar

import shapely
import shapely.strtree
import shapely.wkb

import gws
import gws.lib.osx
from gws.lib.console import ProgressIndicator
import gws.plugin.postgres.provider

import gws.types as t

from . import types as dt
from . import index
from . import norbit6

from .geo_info_dok import gid6 as gid


def run(ix: index.Object, data_schema: str, with_force=False, with_cache=False):
    if with_force:
        ix.drop()
    elif ix.exists():
        gws.log.info('ALKIS index ok')
        return

    rdr = norbit6.Object(ix.provider, schema=data_schema)
    rr = _Runner(ix, rdr, with_cache)
    rr.run()


##

T = TypeVar("T")


class _ObjectDict(Generic[T]):
    def __init__(self, cls):
        self.d = {}
        self.cls = cls

    def add(self, uid, recs) -> T:
        o = self.cls(uid=uid, recs=recs)
        o.isHistoric = all(r.isHistoric for r in recs)
        self.d[o.uid] = o
        return o

    def get(self, uid, default=None) -> t.Optional[T]:
        return self.d.get(uid, default)

    def get_many(self, uids) -> list[T]:
        res = {}

        for uid in uids:
            if uid not in res:
                o = self.d.get(uid)
                if o:
                    res[uid] = o

        return list(res.values())

    def get_from_ptr(self, obj: dt.Entity, attr):
        uids = []

        for r in obj.recs:
            v = _pop(r, attr)
            if isinstance(v, list):
                uids.extend(v)
            elif isinstance(v, str):
                uids.append(v)

        return self.get_many(uids)

    def __iter__(self) -> t.Iterable[T]:
        yield from self.d.values()

    def __len__(self):
        return len(self.d)


class _ObjectMap:

    def __init__(self):
        self.Anschrift: _ObjectDict[dt.Anschrift] = _ObjectDict(dt.Anschrift)
        self.Buchungsblatt: _ObjectDict[dt.Buchungsblatt] = _ObjectDict(dt.Buchungsblatt)
        self.Buchungsstelle: _ObjectDict[dt.Buchungsstelle] = _ObjectDict(dt.Buchungsstelle)
        self.Flurstueck: _ObjectDict[dt.Flurstueck] = _ObjectDict(dt.Flurstueck)
        self.Gebaeude: _ObjectDict[dt.Gebaeude] = _ObjectDict(dt.Gebaeude)
        self.Lage: _ObjectDict[dt.Lage] = _ObjectDict(dt.Lage)
        self.Namensnummer: _ObjectDict[dt.Namensnummer] = _ObjectDict(dt.Namensnummer)
        self.Part: _ObjectDict[dt.Part] = _ObjectDict(dt.Part)
        self.Person: _ObjectDict[dt.Person] = _ObjectDict(dt.Person)

        self.placeAll: dict = {}
        self.placeIdx: dict = {}
        self.catalog: dict = {}


class _Indexer:
    CACHE_KEY: str = ''

    def __init__(self, runner: '_Runner'):
        self.rr = runner
        self.ix: index.Object = runner.ix
        self.om = _ObjectMap()

    def load_or_collect(self):
        if not self.load_cache():
            self.collect()
            self.store_cache()

    def load_cache(self):
        if not self.rr.withCache or not self.CACHE_KEY:
            return False
        cpath = self.rr.cacheDir + '/' + self.CACHE_KEY
        if not gws.is_file(cpath):
            return False
        om = gws.unserialize_from_path(cpath)
        if not om:
            return False
        gws.log.info(f'ALKIS: use cache {self.CACHE_KEY!r}')
        self.om = om
        return True

    def store_cache(self):
        if not self.rr.withCache or not self.CACHE_KEY:
            return
        cpath = self.rr.cacheDir + '/' + self.CACHE_KEY
        gws.serialize_to_path(self.om, cpath)
        gws.log.info(f'ALKIS: store cache {self.CACHE_KEY!r}')

    def collect(self):
        pass

    def write_table(self, table_id, values):
        if self.ix.has_table(table_id):
            return
        with ProgressIndicator(f'ALKIS: write {table_id!r}', len(values)) as progress:
            self.ix.create_table(table_id, values, progress)

    def write(self):
        pass


class _PlaceIndexer(_Indexer):
    """Index places (Administration- und Verwaltungseinheiten).

    References: https://de.wikipedia.org/wiki/Amtlicher_Gemeindeschl%C3%BCssel
    """

    CACHE_KEY = index.TABLE_PLACE

    empty1 = dt.EnumPair(code='0', text='')
    empty2 = dt.EnumPair(code='00', text='')

    def add(self, kind, ax, key_obj, **kwargs):
        if ax.endet:
            return

        code = self.code(kind, key_obj)
        value = dt.EnumPair(code, ax.bezeichnung)

        p = dt.Place(**kwargs)

        p.uid = kind + code
        p.kind = kind
        setattr(p, kind, value)

        self.om.placeAll[p.uid] = p
        self.om.placeIdx[p.uid] = value

        return value

    def collect(self):
        self.om.placeAll = {}
        self.om.placeIdx = {}

        for ax in self.rr.read_flat(gid.AX_Bundesland):
            self.add('land', ax, ax.schluessel)

        for ax in self.rr.read_flat(gid.AX_Regierungsbezirk):
            o = ax.schluessel
            self.add('regierungsbezirk', ax, o, land=self.get_land(o))

        for ax in self.rr.read_flat(gid.AX_KreisRegion):
            o = ax.schluessel
            self.add('kreis', ax, o, land=self.get_land(o), regierungsbezirk=self.get_regierungsbezirk(o))

        for ax in self.rr.read_flat(gid.AX_Gemeinde):
            o = ax.gemeindekennzeichen
            self.add('gemeinde', ax, o, land=self.get_land(o), regierungsbezirk=self.get_regierungsbezirk(o), kreis=self.get_kreis(o))

        # @TODO map Gemarkung to Gemeinde (see https://de.wikipedia.org/wiki/Liste_der_Gemarkungen_in_Nordrhein-Westfalen etc)

        for ax in self.rr.read_flat(gid.AX_Gemarkung):
            o = ax.schluessel
            self.add('gemarkung', ax, o, land=self.get_land(o))

        for ax in self.rr.read_flat(gid.AX_Buchungsblattbezirk):
            o = ax.schluessel
            self.add('buchungsblattbezirk', ax, o, land=self.get_land(o))

        for ax in self.rr.read_flat(gid.AX_Dienststelle):
            o = ax.schluessel
            self.add('dienststelle', ax, o, land=self.get_land(o))

    def write(self):
        values = []

        for place in self.om.placeAll.values():
            values.append(dict(
                uid=place.uid,
                data=index.serialize(place),
            ))

        self.write_table(index.TABLE_PLACE, values)

    def get_land(self, o):
        return self.get('land', o) or self.empty2

    def get_regierungsbezirk(self, o):
        return self.get('regierungsbezirk', o) or self.empty1

    def get_kreis(self, o):
        return self.get('kreis', o) or self.empty2

    def get_gemeinde(self, o):
        return self.get('gemeinde', o) or self.empty1

    def get_gemarkung(self, o):
        return self.get('gemarkung', o) or self.empty1

    def get_buchungsblattbezirk(self, o):
        return self.get('buchungsblattbezirk', o) or self.empty1

    def get_dienststelle(self, o):
        return self.get('dienststelle', o) or self.empty1

    def get(self, kind, o):
        return self.om.placeIdx.get(kind + self.code(kind, o))

    def is_empty(self, p: dt.EnumPair):
        return p.code == '0' or p.code == '00'

    CODES = {
        'land': lambda o: o.land,
        'regierungsbezirk': lambda o: o.land + (o.regierungsbezirk or '0'),
        'kreis': lambda o: o.land + (o.regierungsbezirk or '0') + o.kreis,
        'gemeinde': lambda o: o.land + (o.regierungsbezirk or '0') + o.kreis + o.gemeinde,
        'gemarkung': lambda o: o.land + o.gemarkungsnummer,
        'buchungsblattbezirk': lambda o: o.land + o.bezirk,
        'dienststelle': lambda o: o.land + o.stelle,
    }

    def code(self, kind, o):
        return self.CODES[kind](o)


class _LageIndexer(_Indexer):
    CACHE_KEY = index.TABLE_LAGE

    def collect(self):
        for ax in self.rr.read_flat(gid.AX_LagebezeichnungKatalogeintrag):
            self.om.catalog[self.lage_key(ax.schluessel)] = ax.bezeichnung

        for cls in (gid.AX_LagebezeichnungMitHausnummer, gid.AX_LagebezeichnungOhneHausnummer):
            for uid, axs in self.rr.read_grouped(cls):
                self.om.Lage.add(uid, [
                    _from_ax(
                        dt.LageRecord,
                        ax,
                        strasse=self.strasse(ax),
                        hausnummer=index.normalize_hausnummer(ax.hausnummer),
                    )
                    for ax in axs
                ])

        atts = _attributes(gid.METADATA['AX_Gebaeude'], dt.Gebaeude.PROP_KEYS)

        for uid, axs in self.rr.read_grouped(gid.AX_Gebaeude):
            self.om.Gebaeude.add(uid, [
                _from_ax(
                    dt.GebaeudeRecord,
                    ax,
                    name=', '.join(ax.name) if ax.name else None,
                    amtlicheFlaeche=ax.grundflaeche or 0,
                    props=self.rr.props_from(ax, atts),
                    _zeigtAuf=ax.zeigtAuf,
                )
                for ax in axs
            ])

        for ge in self.om.Gebaeude:
            for r in ge.recs:
                geom = _geom_of(r)
                r.geomFlaeche = round(geom.area, 2) if geom else 0

        # omit Gebaeude geometries for now
        for ge in self.om.Gebaeude:
            for r in ge.recs:
                _pop(r, 'geom')

        for la in self.om.Lage:
            la.gebaeudeList = []

        # AX_Gebaeude.zeigtAuf -> AX_LagebezeichnungMitHausnummer
        for ge in self.om.Gebaeude:
            for la in self.om.Lage.get_from_ptr(ge, '_zeigtAuf'):
                la.gebaeudeList.append(ge)

    def strasse(self, ax):
        if isinstance(ax.lagebezeichnung, str):
            return ax.lagebezeichnung
        return self.om.catalog.get(self.lage_key(ax.lagebezeichnung), '')

    def lage_key(self, r):
        return _comma([
            getattr(r, 'land'),
            getattr(r, 'regierungsbezirk'),
            getattr(r, 'kreis'),
            getattr(r, 'gemeinde'),
            getattr(r, 'lage'),
        ])

    def write(self):
        values = []

        for la in self.om.Lage:
            values.append(dict(
                uid=la.uid,
                rc=len(la.recs),
                data=index.serialize(la),
            ))

        self.write_table(index.TABLE_LAGE, values)


class _BuchungIndexer(_Indexer):
    CACHE_KEY = index.TABLE_BUCHUNGSBLATT

    buchungsblattkennzeichenMap: dict[str, dt.Buchungsblatt] = {}

    def collect(self):
        for uid, axs in self.rr.read_grouped(gid.AX_Anschrift):
            self.om.Anschrift.add(uid, [
                _from_ax(
                    dt.AnschriftRecord,
                    ax,
                    ort=ax.ort_AmtlichesOrtsnamensverzeichnis or ax.ort_Post,
                    plz=ax.postleitzahlPostzustellung,
                    telefon=ax.telefon[0] if ax.telefon else None
                )
                for ax in axs
            ])

        for uid, axs in self.rr.read_grouped(gid.AX_Person):
            self.om.Person.add(uid, [
                _from_ax(
                    dt.PersonRecord,
                    ax,
                    anrede=ax.anrede.text if ax.anrede else None,
                    _hat=ax.hat,
                )
                for ax in axs
            ])

        # AX_Person.hat -> [AX_Anschrift]
        for pe in self.om.Person:
            pe.anschriftList = self.om.Anschrift.get_from_ptr(pe, '_hat')

        for uid, axs in self.rr.read_grouped(gid.AX_Namensnummer):
            self.om.Namensnummer.add(uid, [
                _from_ax(
                    dt.NamensnummerRecord,
                    ax,
                    anteil=_anteil(ax),
                    _benennt=ax.benennt,
                    _istBestandteilVon=ax.istBestandteilVon
                )
                for ax in axs
            ])

        # AX_Namensnummer.benennt -> AX_Person
        for nn in self.om.Namensnummer:
            nn.laufendeNummer = nn.recs[-1].laufendeNummerNachDIN1421
            nn.personList = self.om.Person.get_from_ptr(nn, '_benennt')

        for uid, axs in self.rr.read_grouped(gid.AX_Buchungsstelle):
            self.om.Buchungsstelle.add(uid, [
                _from_ax(
                    dt.BuchungsstelleRecord,
                    ax,
                    anteil=_anteil(ax),
                    _an=ax.an,
                    _zu=ax.zu,
                    _istBestandteilVon=ax.istBestandteilVon,
                )
                for ax in axs
            ])

        for bs in self.om.Buchungsstelle:
            bs.laufendeNummer = bs.recs[-1].laufendeNummer
            bs.fsUids = []
            bs.flurstueckskennzeichenList = []

        for uid, axs in self.rr.read_grouped(gid.AX_Buchungsblatt):
            self.om.Buchungsblatt.add(uid, [
                _from_ax(
                    dt.BuchungsblattRecord,
                    ax,
                    buchungsblattbezirk=self.rr.place.get_buchungsblattbezirk(ax.buchungsblattbezirk),
                )
                for ax in axs
            ])

        for bb in self.om.Buchungsblatt:
            bb.buchungsstelleList = []
            bb.namensnummerList = []
            bb.buchungsblattkennzeichen = bb.recs[-1].buchungsblattkennzeichen
            self.buchungsblattkennzeichenMap[bb.buchungsblattkennzeichen] = bb

        # AX_Buchungsstelle.istBestandteilVon -> AX_Buchungsblatt
        for bs in self.om.Buchungsstelle:
            bb_list = self.om.Buchungsblatt.get_from_ptr(bs, '_istBestandteilVon')
            bs.buchungsblattUids = [bb.uid for bb in bb_list]
            bs.buchungsblattkennzeichenList = [bb.buchungsblattkennzeichen for bb in bb_list]
            for bb in bb_list:
                bb.buchungsstelleList.append(bs)

        # AX_Namensnummer.istBestandteilVon -> AX_Buchungsblatt
        for nn in self.om.Namensnummer:
            bb_list = self.om.Buchungsblatt.get_from_ptr(nn, '_istBestandteilVon')
            nn.buchungsblattUids = [bb.uid for bb in bb_list]
            nn.buchungsblattkennzeichenList = [bb.buchungsblattkennzeichen for bb in bb_list]
            for bb in bb_list:
                bb.namensnummerList.append(nn)

        for bb in self.om.Buchungsblatt:
            bb.buchungsstelleList.sort(key=_sortkey_buchungsstelle)
            bb.namensnummerList.sort(key=_sortkey_namensnummer)

        # AX_Buchungsstelle.an -> [AX_Buchungsstelle]
        # AX_Buchungsstelle.zu -> [AX_Buchungsstelle]
        # see Erläuterungen zu ALKIS Version 6, page 116-119

        for bs in self.om.Buchungsstelle:
            bs.childUids = []
            bs.parentUids = []
            bs.parentkennzeichenList = []

        for bs in self.om.Buchungsstelle:
            parent_uids = set()
            parent_knz = set()

            for r in bs.recs:
                parent_uids.update(_pop(r, '_an'))
                parent_uids.update(_pop(r, '_zu'))

            for parent_bs in self.om.Buchungsstelle.get_many(parent_uids):
                parent_bs.childUids.append(bs.uid)
                bs.parentUids.append(parent_bs.uid)
                for bb_knz in parent_bs.buchungsblattkennzeichenList:
                    parent_knz.add(bb_knz + '.' + parent_bs.laufendeNummer)

            bs.parentkennzeichenList = sorted(parent_knz)

    def write(self):
        values = []

        for bb in self.om.Buchungsblatt:
            values.append(dict(
                uid=bb.uid,
                rc=len(bb.recs),
                data=index.serialize(bb),
            ))

        self.write_table(index.TABLE_BUCHUNGSBLATT, values)


class _PartIndexer(_Indexer):
    CACHE_KEY = index.TABLE_PART
    MIN_PART_AREA = 1

    parts: list[dt.Part] = []

    fs_list = []
    fs_geom = []

    stree: shapely.strtree.STRtree

    def collect(self):

        for kind in dt.Part.KIND:
            self.collect_kind(kind)

        for fs in self.rr.fsdata.om.Flurstueck:
            self.fs_list.append(fs)
            # NB take only the most recent fs geometry into account
            self.fs_geom.append(_geom_of(fs.recs[-1]))

        self.stree = shapely.strtree.STRtree(self.fs_geom)

        with ProgressIndicator(f'ALKIS: parts', len(self.om.Part)) as progress:
            for pa in self.om.Part:
                self.compute_intersections(pa)
                progress.update(1)

        self.parts.sort(key=_sortkey_part)

        for pa in self.parts:
            pa.isHistoric = all(r.isHistoric for r in pa.recs)

    def collect_kind(self, kind):
        _, key = dt.Part.KIND[kind]
        classes = [
            getattr(gid, meta['name'])
            for meta in gid.METADATA.values()
            if (
                    meta['kind'] == 'object'
                    and meta['geom']
                    and re.search(key + r'/\w+/', meta['key'])
            )
        ]

        for cls in classes:
            self.collect_class(kind, cls)

    def collect_class(self, kind, cls):
        meta = gid.METADATA[cls.__name__]
        atts = _attributes(meta, dt.Part.PROP_KEYS)

        for uid, axs in self.rr.read_grouped(cls):
            pa = self.om.Part.add(uid, [
                _from_ax(
                    dt.PartRecord,
                    ax,
                    props=self.rr.props_from(ax, atts),
                )
                for ax in axs
            ])
            pa.kind = kind
            pa.name = dt.EnumPair(meta['uid'], meta['title'])

    def compute_intersections(self, pa: dt.Part):
        parts_map = {}

        for r in pa.recs:
            geom = _geom_of(r)
            if not geom:
                continue

            for i in self.stree.query(geom):
                part_geom = shapely.intersection(self.fs_geom[i], geom)
                part_area = round(part_geom.area, 2)
                if part_area < self.MIN_PART_AREA:
                    continue

                fs = self.fs_list[i]

                part = parts_map.setdefault(fs.uid, dt.Part(
                    uid=pa.uid,
                    recs=[],
                    kind=pa.kind,
                    name=pa.name,
                    fs=fs.uid,
                ))

                # computed area corrected with respect to FS's "amtlicheFlaeche"
                part_area_corrected = round(
                    fs.recs[-1].amtlicheFlaeche * (part_area / fs.recs[-1].geomFlaeche),
                    2)

                part.recs.append(dt.PartRecord(
                    uid=r.uid,
                    beginnt=r.beginnt,
                    endet=r.endet,
                    anlass=r.anlass,
                    props=r.props,
                    geomFlaeche=part_area,
                    amtlicheFlaeche=part_area_corrected,
                    isHistoric=r.endet is not None,
                ))

                part.geom = shapely.wkb.dumps(part_geom, srid=self.ix.crs.srid, hex=True)
                part.geomFlaeche = part_area
                part.amtlicheFlaeche = part_area_corrected

        self.parts.extend(parts_map.values())

    def write(self):
        values = []

        for n, pa in enumerate(self.parts, 1):
            geom = _pop(pa, 'geom')
            data = index.serialize(pa)
            pa.geom = geom
            values.append(dict(
                n=n,
                fs=pa.fs,
                uid=pa.uid,
                beginnt=pa.recs[-1].beginnt,
                endet=pa.recs[-1].endet,
                kind=pa.kind,
                name=pa.name.text,
                parthistoric=pa.isHistoric,
                data=data,
                geom=geom,
            ))

        self.write_table(index.TABLE_PART, values)


class _FsDataIndexer(_Indexer):
    CACHE_KEY = index.TABLE_FLURSTUECK

    def collect(self):
        for uid, axs in self.rr.read_grouped(gid.AX_Flurstueck):
            recs = gws.compact(self.record(ax) for ax in axs)
            if recs:
                self.om.Flurstueck.add(uid, recs)

        for uid, axs in self.rr.read_grouped(gid.AX_HistorischesFlurstueck):
            recs = gws.compact(self.record(ax) for ax in axs)
            if not recs:
                continue
            # For a historic FS, 'beginnt' is basically when the history beginnt
            # (see comments for AX_HistorischesFlurstueck in gid6).
            # we set F.endet=F.beginnt to designate this one as 'historic'
            for r in recs:
                r.endet = r.beginnt
                r.isHistoric = True
            self.om.Flurstueck.add(uid, recs)

        for fs in self.om.Flurstueck:
            fs.flurstueckskennzeichen = fs.recs[-1].flurstueckskennzeichen
            self.process_lage(fs)
            self.process_gebaeude(fs)
            self.process_buchung(fs)

    def record(self, ax):
        r: dt.FlurstueckRecord = _from_ax(
            dt.FlurstueckRecord,
            ax,
            amtlicheFlaeche=ax.amtlicheFlaeche or 0,
            flurnummer=_str(ax.flurnummer),
            zaehler=_str(ax.flurstuecksnummer.zaehler),
            nenner=_str(ax.flurstuecksnummer.nenner),
            zustaendigeStelle=[self.rr.place.get_dienststelle(p) for p in (ax.zustaendigeStelle or [])],

            _weistAuf=ax.weistAuf,
            _zeigtAuf=ax.zeigtAuf,
            _istGebucht=ax.istGebucht,
            _buchung=ax.buchung,
        )

        # basic data

        r.gemarkung = self.rr.place.get_gemarkung(ax.gemarkung)
        r.gemeinde = self.rr.place.get_gemeinde(ax.gemeindezugehoerigkeit)
        r.regierungsbezirk = self.rr.place.get_regierungsbezirk(ax.gemeindezugehoerigkeit)
        r.kreis = self.rr.place.get_kreis(ax.gemeindezugehoerigkeit)
        r.land = self.rr.place.get_land(ax.gemeindezugehoerigkeit)

        if self.rr.place.is_empty(r.gemarkung) or self.rr.place.is_empty(r.gemeinde):
            # exclude Flurstücke that refer to Gemeinde/Gemarkung
            # which do not exist in the reference AX tables
            return None

        if r.gemarkung.code in self.ix.excludeGemarkung:
            return None

        # geometry

        geom = _geom_of(r)
        if not geom:
            return None

        r.geomFlaeche = round(geom.area, 2)
        r.x = round(geom.centroid.x, 2)
        r.y = round(geom.centroid.y, 2)

        return r

    def process_lage(self, fs: dt.Flurstueck):
        fs.lageList = []

        # AX_Flurstueck.weistAuf -> AX_LagebezeichnungMitHausnummer
        # AX_Flurstueck.zeigtAuf -> AX_LagebezeichnungOhneHausnummer
        fs.lageList.extend(self.rr.lage.om.Lage.get_from_ptr(fs, '_weistAuf'))
        fs.lageList.extend(self.rr.lage.om.Lage.get_from_ptr(fs, '_zeigtAuf'))

    def process_gebaeude(self, fs: dt.Flurstueck):
        ge_map = {}

        for la in fs.lageList:
            for ge in la.gebaeudeList:
                ge_map[ge.uid] = ge

        fs.gebaeudeList = list(ge_map.values())
        fs.gebaeudeList.sort(key=_sortkey_gebaeude)

        fs.gebaeudeAmtlicheFlaeche = sum(ge.recs[-1].amtlicheFlaeche for ge in fs.gebaeudeList if not ge.recs[-1].endet)
        fs.gebaeudeGeomFlaeche = sum(ge.recs[-1].geomFlaeche for ge in fs.gebaeudeList if not ge.recs[-1].endet)

    def process_buchung(self, fs: dt.Flurstueck):
        bs_historic_map = {}
        bs_seen = set()
        buchung_map = {}

        # for each Flurstück record, we collect all related Buchungsstellen (with respect to parent-child relations)
        # then group Buchungsstellen by their Buchungsblatt
        # and create Buchung objects for a FS

        for r in fs.recs:
            hist_buchung = _pop(r, '_buchung')
            if hist_buchung:
                bs_list = self.historic_buchungsstelle_list(r, hist_buchung)
            else:
                bs_list = self.buchungsstelle_list(r)

            for bs in bs_list:
                # a Buchungsstelle referred to by an expired Flurstück might not be expired itself,
                # so we have to track its state separately by wrapping it in a BuchungsstelleReference
                # a BuchungsstelleReference is historic if its Flurstück is
                bs_historic_map[bs.uid] = r.isHistoric

                if bs.uid in bs_seen:
                    continue
                bs_seen.add(bs.uid)

                # populate Flurstück references in a Buchungsstelle
                if fs.uid not in bs.fsUids:
                    bs.fsUids.append(fs.uid)
                    bs.flurstueckskennzeichenList.append(fs.flurstueckskennzeichen)

                # create Buchung records by grouping Buchungsstellen
                for bb_uid in bs.buchungsblattUids:
                    bu = buchung_map.setdefault(bb_uid, dt.Buchung(recs=[], buchungsblattUid=bb_uid))
                    bu.recs.append(dt.BuchungsstelleReference(buchungsstelle=bs))

        fs.buchungList = list(buchung_map.values())

        for bu in fs.buchungList:
            for ref in bu.recs:
                ref.isHistoric = bs_historic_map[ref.buchungsstelle.uid]
            bu.isHistoric = all(ref.isHistoric for ref in bu.recs)

        return fs

    def historic_buchungsstelle_list(self, r: dt.FlurstueckRecord, hist_buchung):
        # an AX_HistorischesFlurstueck with a speicial 'buchung' reference

        bs_list = []

        for bu in hist_buchung:
            bb = self.rr.buchung.buchungsblattkennzeichenMap.get(bu.buchungsblattkennzeichen)
            if not bb:
                continue
            # create a fake historic Buchungstelle
            bs_list.append(dt.Buchungsstelle(
                uid=bb.uid + '_' + bu.laufendeNummerDerBuchungsstelle,
                recs=[
                    dt.BuchungsstelleRecord(
                        endet=r.endet,
                        laufendeNummer=bu.laufendeNummerDerBuchungsstelle,
                        isHistoric=True,
                    )
                ],
                buchungsblattUids=[bb.uid],
                buchungsblattkennzeichenList=[bb.buchungsblattkennzeichen],
                parentUids=[],
                childUids=[],
                fsUids=[],
                parentkennzeichenList=[],
                flurstueckskennzeichenList=[],
                laufendeNummer=bu.laufendeNummerDerBuchungsstelle,
                isHistoric=True,
            ))

        return bs_list

    def buchungsstelle_list(self, r: dt.FlurstueckRecord):
        # AX_Flurstueck.istGebucht -> AX_Buchungsstelle

        this_bs = self.rr.buchung.om.Buchungsstelle.get(_pop(r, '_istGebucht'))
        if not this_bs:
            return []

        bs_list = []

        # A Flurstück points to a Buchungsstelle (F.istGebucht -> B).
        # A Buchungsstelle can have parent (B.an -> parent.uid) and child (child.an -> B.uid) Buchungsstellen
        # (these references are populated in _BuchungIndexer above).
        # Our task here, given F.istGebucht -> B, collect B's parents and children
        # These are Buchungsstellen that directly or indirectly mention the current Flurstück.

        queue: list[dt.Buchungsstelle] = [this_bs]
        while queue:
            bs = queue.pop(0)
            bs_list.insert(0, bs)
            for uid in bs.parentUids:
                queue.append(self.rr.buchung.om.Buchungsstelle.get(uid))

        # remove this_bs
        bs_list.pop()

        queue: list[dt.Buchungsstelle] = [this_bs]
        while queue:
            bs = queue.pop(0)
            bs_list.append(bs)
            # sort related (child) Buchungsstellen by their BB-Kennzeichen
            child_bs_list = self.rr.buchung.om.Buchungsstelle.get_many(bs.childUids)
            child_bs_list.sort(key=_sortkey_buchungsstelle_by_bblatt)
            queue.extend(child_bs_list)

        # if len(bs_list) > 1:
        #     gws.log.debug(f'bs chain: {r.uid=} {this_bs.uid=} {[bs.uid for bs in bs_list]}')

        return bs_list

    def write(self):
        values = []

        for fs in self.om.Flurstueck:
            geoms = [_pop(r, 'geom') for r in fs.recs]
            data = index.serialize(fs)
            for r, g in zip(fs.recs, geoms):
                r.geom = g

            values.append(dict(
                uid=fs.uid,
                rc=len(fs.recs),
                fshistoric=fs.isHistoric,
                data=data,
                geom=geoms[-1],
            ))

        self.write_table(index.TABLE_FLURSTUECK, values)


class _FsIndexIndexer(_Indexer):
    entries = {
        index.TABLE_INDEXFLURSTUECK: [],
        index.TABLE_INDEXLAGE: [],
        index.TABLE_INDEXBUCHUNGSBLATT: [],
        index.TABLE_INDEXPERSON: [],
        index.TABLE_INDEXGEOM: [],
    }

    def collect(self):
        with ProgressIndicator(f'ALKIS: creating indexes', len(self.rr.fsdata.om.Flurstueck)) as progress:
            for fs in self.rr.fsdata.om.Flurstueck:
                for r in fs.recs:
                    self.add(fs, r)
                progress.update(1)

    def add(self, fs: dt.Flurstueck, r: dt.FlurstueckRecord):
        base = dict(
            fs=r.uid,
            fshistoric=r.isHistoric,
        )

        places = dict(
            land=r.land.text,
            land_t=index.text_key(r.land.text),
            landcode=r.land.code,

            regierungsbezirk=r.regierungsbezirk.text,
            regierungsbezirk_t=index.text_key(r.regierungsbezirk.text),
            regierungsbezirkcode=r.regierungsbezirk.code,

            kreis=r.kreis.text,
            kreis_t=index.text_key(r.kreis.text),
            kreiscode=r.kreis.code,

            gemeinde=r.gemeinde.text,
            gemeinde_t=index.text_key(r.gemeinde.text),
            gemeindecode=r.gemeinde.code,

            gemarkung=r.gemarkung.text,
            gemarkung_t=index.text_key(r.gemarkung.text),
            gemarkungcode=r.gemarkung.code,

        )

        self.entries[index.TABLE_INDEXFLURSTUECK].append(dict(
            **base,
            **places,

            amtlicheflaeche=r.amtlicheFlaeche,
            geomflaeche=r.geomFlaeche,

            flurnummer=r.flurnummer,
            zaehler=r.zaehler,
            nenner=r.nenner,
            flurstuecksfolge=r.flurstuecksfolge,
            flurstueckskennzeichen=r.flurstueckskennzeichen,

            x=r.x,
            y=r.y,
        ))

        self.entries[index.TABLE_INDEXGEOM].append(dict(
            **base,
            geomflaeche=r.geomFlaeche,
            x=r.x,
            y=r.y,
            geom=r.geom,
        ))

        for la in fs.lageList:
            for la_r in la.recs:
                self.entries[index.TABLE_INDEXLAGE].append(dict(
                    **base,
                    **places,
                    lageuid=la_r.uid,
                    lagehistoric=la_r.isHistoric,
                    strasse=la_r.strasse,
                    strasse_t=index.strasse_key(la_r.strasse),
                    hausnummer=la_r.hausnummer,
                    x=r.x,
                    y=r.y,
                ))

        for bu in fs.buchungList:
            bb = self.rr.buchung.om.Buchungsblatt.get(bu.buchungsblattUid)

            for bb_r in bb.recs:
                self.entries[index.TABLE_INDEXBUCHUNGSBLATT].append(dict(
                    **base,
                    buchungsblattuid=bb_r.uid,
                    buchungsblattkennzeichen=bb_r.buchungsblattkennzeichen,
                    buchungsblatthistoric=bu.isHistoric,
                ))

            pe_uids = set()

            for nn in bb.namensnummerList:
                for pe in nn.personList:
                    if pe.uid in pe_uids:
                        continue
                    pe_uids.add(pe.uid)
                    for pe_r in pe.recs:
                        self.entries[index.TABLE_INDEXPERSON].append(dict(
                            **base,
                            personuid=pe_r.uid,
                            personhistoric=pe_r.isHistoric,
                            name=pe_r.nachnameOderFirma,
                            name_t=index.text_key(pe_r.nachnameOderFirma),
                            vorname=pe_r.vorname,
                            vorname_t=index.text_key(pe_r.nachnameOderFirma),
                        ))

    def write(self):
        for table_id, values in self.entries.items():
            if not self.ix.has_table(table_id):
                for n, v in enumerate(values, 1):
                    v['n'] = n
                self.write_table(table_id, values)


class _Runner:
    def __init__(self, ix: index.Object, reader: dt.Reader, with_cache=False):
        self.ix: index.Object = ix
        self.reader: dt.Reader = reader

        self.withCache = with_cache
        self.cacheDir = gws.CACHE_DIR + '/alkis'
        if self.withCache:
            gws.ensure_dir(self.cacheDir)

        self.place = _PlaceIndexer(self)
        self.lage = _LageIndexer(self)
        self.buchung = _BuchungIndexer(self)
        self.part = _PartIndexer(self)
        self.fsdata = _FsDataIndexer(self)
        self.fsindex = _FsIndexIndexer(self)

        self.initMemory = gws.lib.osx.process_rss_size()

    def run(self):
        with ProgressIndicator(f'ALKIS: indexing'):
            self.place.load_or_collect()
            self.memory_info()

            self.buchung.load_or_collect()
            self.memory_info()

            self.lage.load_or_collect()
            self.memory_info()

            self.fsdata.load_or_collect()
            self.memory_info()

            self.part.load_or_collect()
            self.memory_info()

            self.fsindex.collect()
            self.memory_info()

            self.place.write()
            self.buchung.write()
            self.lage.write()
            self.fsdata.write()
            self.part.write()
            self.fsindex.write()

    def memory_info(self):
        v = gws.lib.osx.process_rss_size() - self.initMemory
        if v > 0:
            gws.log.info(f'ALKIS: memory used: {v:.2f} MB', stacklevel=2)

    def read_flat(self, cls):
        cpath = self.cacheDir + '/flat_' + cls.__name__
        if self.withCache and gws.is_file(cpath):
            return gws.unserialize_from_path(cpath)

        rs = self._read_flat(cls)
        if self.withCache:
            gws.serialize_to_path(rs, cpath)

        return rs

    def _read_flat(self, cls):
        cnt = self.reader.count(cls)
        if cnt <= 0:
            return []

        rs = []
        with ProgressIndicator(f'ALKIS: read {cls.__name__}', cnt) as progress:
            for ax in self.reader.read_all(cls):
                rs.append(ax)
                progress.update(1)
        return rs

    def read_grouped(self, cls):
        cpath = self.cacheDir + '/grouped_' + cls.__name__
        if self.withCache and gws.is_file(cpath):
            return gws.unserialize_from_path(cpath)

        rs = self._read_grouped(cls)
        if self.withCache:
            gws.serialize_to_path(rs, cpath)

        return rs

    def _read_grouped(self, cls):
        cnt = self.reader.count(cls)
        if cnt <= 0:
            return []

        groups = {}
        with ProgressIndicator(f'ALKIS: read {cls.__name__}', cnt) as progress:
            for ax in self.reader.read_all(cls):
                groups.setdefault(ax.identifikator, []).append(ax)
                progress.update(1)
        for g in groups.values():
            g.sort(key=_sortkey_lebenszeitintervall)

        return list(groups.items())

    def props_from(self, ax, atts):
        props = []

        for a in atts:
            v = getattr(ax, a['name'], None)
            if isinstance(v, gid.Object):
                v = self.object_prop_value(v)
            if not gws.is_empty(v):
                props.append([a['title'], v])

        return props

    def object_prop_value(self, o):
        return None
        # @TODO handle properties which are objects
        # gid.AX_Bundesland_Schluessel
        # gid.AX_Dienststelle_Schluessel
        # gid.AX_Gemarkung_Schluessel
        # gid.AX_Gemeindekennzeichen
        # gid.AX_Kreis_Schluessel
        # gid.AX_Regierungsbezirk_Schluessel
        # gid.AX_KennzifferGrabloch
        # gid.AX_Lagebezeichnung
        # gid.AX_Tagesabschnitt
        # gid.AX_Verwaltungsgemeinschaft_Schluessel


def _from_ax(cls, ax, **kwargs):
    d = {}

    if ax:
        for k in cls.__annotations__:
            v = getattr(ax, k, None)
            if v:
                d[k] = v

        d['uid'] = ax.identifikator
        d['beginnt'] = ax.lebenszeitintervall.beginnt
        d['endet'] = ax.lebenszeitintervall.endet
        d['isHistoric'] = d['endet'] is not None
        if ax.anlass and ax.anlass[0].code != '000000':
            d['anlass'] = ax.anlass[0]
        if ax.geom:
            d['geom'] = ax.geom

    d.update(kwargs)
    return cls(**d)


def _anteil(ax):
    try:
        z = float(ax.anteil.zaehler)
        z = str(int(z) if z.is_integer() else z)
        n = float(ax.anteil.nenner)
        n = str(int(n) if n.is_integer() else n)
        return z + '/' + n
    except (AttributeError, ValueError, TypeError):
        pass


def _attributes(meta, keys):
    return sorted(
        [a for a in meta['attributes'] if a['name'] in keys],
        key=lambda a: a['title']
    )


def _geom_of(o):
    if not o.geom:
        gws.log.warning(f'{o.__class__.__name__}:{o.uid}: no geometry')
        return
    return shapely.wkb.loads(o.geom, hex=True)


def _pop(obj, attr):
    v = getattr(obj, attr, None)
    try:
        delattr(obj, attr)
    except AttributeError:
        pass
    return v


def _sortkey_beginnt(o):
    return o.beginnt


def _sortkey_lebenszeitintervall(o):
    return o.lebenszeitintervall.beginnt


def _sortkey_namensnummer(nn: dt.Namensnummer):
    return _natkey(nn.recs[-1].laufendeNummerNachDIN1421), nn.recs[-1].beginnt


def _sortkey_buchungsstelle(bs: dt.Buchungsstelle):
    return _natkey(bs.recs[-1].laufendeNummer), bs.recs[-1].beginnt


def _sortkey_buchungsstelle_by_bblatt(bs: dt.Buchungsstelle):
    return bs.buchungsblattkennzeichenList[0], bs.recs[-1].beginnt


def _sortkey_part(pa: dt.Part):
    return pa.name.text, -pa.geomFlaeche


def _sortkey_gebaeude(ge: dt.Gebaeude):
    # sort Gebaeude by area (big->small)
    return ge.recs[-1].beginnt, -ge.recs[-1].geomFlaeche


def _natkey(v):
    if not v:
        return []
    return [
        int(a) if a else b.lower()
        for a, b in re.findall(r'(\d+)|(\D+)', v.strip())
    ]


def _comma(a):
    return ','.join(str(s) if s is not None else '' for s in a)


def _str(x):
    return None if x is None else str(x)
