from typing import Optional, Iterable

import re
import datetime

from sqlalchemy.dialects.postgresql import JSONB

import gws
import gws.base.shape
import gws.base.database
import gws.config.util
import gws.lib.crs
import gws.plugin.postgres.provider
import gws.lib.sa as sa
from gws.lib.cli import ProgressIndicator

from . import types as dt

TABLE_PLACE = 'place'
TABLE_FLURSTUECK = 'flurstueck'
TABLE_BUCHUNGSBLATT = 'buchungsblatt'
TABLE_LAGE = 'lage'
TABLE_PART = 'part'

TABLE_INDEXFLURSTUECK = 'indexflurstueck'
TABLE_INDEXLAGE = 'indexlage'
TABLE_INDEXBUCHUNGSBLATT = 'indexbuchungsblatt'
TABLE_INDEXPERSON = 'indexperson'
TABLE_INDEXGEOM = 'indexgeom'


class Object(gws.Node):
    VERSION = '82'

    TABLES_BASIC = [
        TABLE_PLACE,
        TABLE_FLURSTUECK,
        TABLE_LAGE,
        TABLE_PART,
        TABLE_INDEXFLURSTUECK,
        TABLE_INDEXLAGE,
        TABLE_INDEXGEOM,
    ]

    TABLES_BUCHUNG = [
        TABLE_BUCHUNGSBLATT,
        TABLE_INDEXBUCHUNGSBLATT,
    ]

    TABLES_EIGENTUEMER = [
        TABLE_BUCHUNGSBLATT,
        TABLE_INDEXBUCHUNGSBLATT,
        TABLE_INDEXPERSON,
    ]

    ALL_TABLES = TABLES_BASIC + TABLES_BUCHUNG + TABLES_EIGENTUEMER

    db: gws.plugin.postgres.provider.Object
    crs: gws.Crs
    schema: str
    excludeGemarkung: set[str]

    saMeta: sa.MetaData
    tables: dict[str, sa.Table]

    columnDct = {}

    def __getstate__(self):
        return gws.u.omit(vars(self), 'saMeta')

    def configure(self):
        gws.config.util.configure_database_provider_for(self, ext_type='postgres')
        self.crs = gws.lib.crs.get(self.cfg('crs'))
        self.schema = self.cfg('schema', default='public')
        self.excludeGemarkung = set(self.cfg('excludeGemarkung', default=[]))
        self.saMeta = sa.MetaData(schema=self.schema)
        self.tables = {}

    def activate(self):
        self.saMeta = sa.MetaData(schema=self.schema)
        self.tables = {}

        self.columnDct = {
            TABLE_PLACE: [
                sa.Column('uid', sa.Text, primary_key=True),
                sa.Column('data', JSONB),
            ],
            TABLE_FLURSTUECK: [
                sa.Column('uid', sa.Text, primary_key=True),
                sa.Column('rc', sa.Integer),
                sa.Column('fshistoric', sa.Boolean),
                sa.Column('data', JSONB),
                sa.Column('geom', sa.geo.Geometry(srid=self.crs.srid)),
            ],
            TABLE_BUCHUNGSBLATT: [
                sa.Column('uid', sa.Text, primary_key=True),
                sa.Column('rc', sa.Integer),
                sa.Column('data', JSONB),
            ],
            TABLE_LAGE: [
                sa.Column('uid', sa.Text, primary_key=True),
                sa.Column('rc', sa.Integer),
                sa.Column('data', JSONB),
            ],
            TABLE_PART: [
                sa.Column('n', sa.Integer, primary_key=True),
                sa.Column('fs', sa.Text, index=True),
                sa.Column('uid', sa.Text, index=True),
                sa.Column('beginnt', sa.DateTime),
                sa.Column('endet', sa.DateTime),
                sa.Column('kind', sa.Integer),
                sa.Column('name', sa.Text),
                sa.Column('parthistoric', sa.Boolean),
                sa.Column('data', JSONB),
                sa.Column('geom', sa.geo.Geometry(srid=self.crs.srid)),
            ],
            TABLE_INDEXFLURSTUECK: [
                sa.Column('n', sa.Integer, primary_key=True),

                sa.Column('fs', sa.Text, index=True),
                sa.Column('fshistoric', sa.Boolean, index=True),

                sa.Column('land', sa.Text, index=True),
                sa.Column('land_t', sa.Text, index=True),
                sa.Column('landcode', sa.Text, index=True),

                sa.Column('regierungsbezirk', sa.Text, index=True),
                sa.Column('regierungsbezirk_t', sa.Text, index=True),
                sa.Column('regierungsbezirkcode', sa.Text, index=True),

                sa.Column('kreis', sa.Text, index=True),
                sa.Column('kreis_t', sa.Text, index=True),
                sa.Column('kreiscode', sa.Text, index=True),

                sa.Column('gemeinde', sa.Text, index=True),
                sa.Column('gemeinde_t', sa.Text, index=True),
                sa.Column('gemeindecode', sa.Text, index=True),

                sa.Column('gemarkung', sa.Text, index=True),
                sa.Column('gemarkung_t', sa.Text, index=True),
                sa.Column('gemarkungcode', sa.Text, index=True),

                sa.Column('amtlicheflaeche', sa.Float, index=True),
                sa.Column('geomflaeche', sa.Float, index=True),

                sa.Column('flurnummer', sa.Text, index=True),
                sa.Column('zaehler', sa.Text, index=True),
                sa.Column('nenner', sa.Text, index=True),
                sa.Column('flurstuecksfolge', sa.Text, index=True),
                sa.Column('flurstueckskennzeichen', sa.Text, index=True),

                sa.Column('x', sa.Float, index=True),
                sa.Column('y', sa.Float, index=True),
            ],
            TABLE_INDEXLAGE: [
                sa.Column('n', sa.Integer, primary_key=True),

                sa.Column('fs', sa.Text, index=True),
                sa.Column('fshistoric', sa.Boolean, index=True),

                sa.Column('land', sa.Text, index=True),
                sa.Column('land_t', sa.Text, index=True),
                sa.Column('landcode', sa.Text, index=True),

                sa.Column('regierungsbezirk', sa.Text, index=True),
                sa.Column('regierungsbezirk_t', sa.Text, index=True),
                sa.Column('regierungsbezirkcode', sa.Text, index=True),

                sa.Column('kreis', sa.Text, index=True),
                sa.Column('kreis_t', sa.Text, index=True),
                sa.Column('kreiscode', sa.Text, index=True),

                sa.Column('gemeinde', sa.Text, index=True),
                sa.Column('gemeinde_t', sa.Text, index=True),
                sa.Column('gemeindecode', sa.Text, index=True),

                sa.Column('gemarkung', sa.Text, index=True),
                sa.Column('gemarkung_t', sa.Text, index=True),
                sa.Column('gemarkungcode', sa.Text, index=True),

                sa.Column('lageuid', sa.Text, index=True),
                sa.Column('lagehistoric', sa.Boolean, index=True),

                sa.Column('strasse', sa.Text, index=True),
                sa.Column('strasse_t', sa.Text, index=True),
                sa.Column('hausnummer', sa.Text, index=True),

                sa.Column('x', sa.Float, index=True),
                sa.Column('y', sa.Float, index=True),
            ],
            TABLE_INDEXBUCHUNGSBLATT: [
                sa.Column('n', sa.Integer, primary_key=True),

                sa.Column('fs', sa.Text, index=True),
                sa.Column('fshistoric', sa.Boolean, index=True),

                sa.Column('buchungsblattuid', sa.Text, index=True),
                sa.Column('buchungsblattbeginnt', sa.DateTime, index=True),
                sa.Column('buchungsblattendet', sa.DateTime, index=True),
                sa.Column('buchungsblattkennzeichen', sa.Text, index=True),
                sa.Column('buchungsblatthistoric', sa.Boolean, index=True),
            ],
            TABLE_INDEXPERSON: [
                sa.Column('n', sa.Integer, primary_key=True),

                sa.Column('fs', sa.Text, index=True),
                sa.Column('fshistoric', sa.Boolean, index=True),

                sa.Column('personuid', sa.Text, index=True),
                sa.Column('personhistoric', sa.Boolean, index=True),

                sa.Column('name', sa.Text, index=True),
                sa.Column('name_t', sa.Text, index=True),
                sa.Column('vorname', sa.Text, index=True),
                sa.Column('vorname_t', sa.Text, index=True),
            ],
            TABLE_INDEXGEOM: [
                sa.Column('n', sa.Integer, primary_key=True),

                sa.Column('fs', sa.Text, index=True),
                sa.Column('fshistoric', sa.Boolean, index=True),

                sa.Column('geomflaeche', sa.Float, index=True),
                sa.Column('x', sa.Float, index=True),
                sa.Column('y', sa.Float, index=True),

                sa.Column('geom', sa.geo.Geometry(srid=self.crs.srid)),
            ],
        }

    ##

    def table(self, table_id: str) -> sa.Table:
        if table_id not in self.tables:
            table_name = f'alkis_{self.VERSION}_{table_id}'
            self.tables[table_id] = sa.Table(
                table_name,
                self.saMeta,
                *self.columnDct[table_id],
                schema=self.schema,
            )
        return self.tables[table_id]

    def table_size(self, table_id) -> int:
        sizes = self._table_size_map([table_id])
        return sizes.get(table_id, 0)

    def _table_size_map(self, table_ids):
        d = {}

        with self.db.connect():
            for table_id in table_ids:
                d[table_id] = self.db.count(self.table(table_id))

        return d

    def has_table(self, table_id: str) -> bool:
        return self.table_size(table_id) > 0

    def status(self) -> dt.IndexStatus:
        sizes = self._table_size_map(self.ALL_TABLES)
        s = dt.IndexStatus(
            basic=all(sizes.get(tid, 0) > 0 for tid in self.TABLES_BASIC),
            buchung=all(sizes.get(tid, 0) > 0 for tid in self.TABLES_BUCHUNG),
            eigentuemer=all(sizes.get(tid, 0) > 0 for tid in self.TABLES_EIGENTUEMER),
        )
        s.complete = s.basic and s.buchung and s.eigentuemer
        s.missing = all(v == 0 for v in sizes.values())
        gws.log.info(f'ALKIS: table sizes {sizes!r}')
        return s

    def drop_table(self, table_id: str):
        with self.db.connect() as conn:
            self._drop_table(conn, table_id)
            conn.commit()

    def drop(self):
        with self.db.connect() as conn:
            for table_id in self.ALL_TABLES:
                self._drop_table(conn, table_id)
            conn.commit()

    def _drop_table(self, conn, table_id):
        tab = self.table(table_id)
        conn.execute(sa.text(f'DROP TABLE IF EXISTS {self.schema}.{tab.name}'))

    INSERT_SIZE = 5000

    def create_table(
            self,
            table_id: str,
            values: list[dict],
            progress: Optional[ProgressIndicator] = None
    ):
        tab = self.table(table_id)
        self.saMeta.create_all(self.db.engine(), tables=[tab])

        with self.db.connect() as conn:
            for i in range(0, len(values), self.INSERT_SIZE):
                vals = values[i:i + self.INSERT_SIZE]
                conn.execute(sa.insert(tab).values(vals))
                conn.commit()
                if progress:
                    progress.update(len(vals))

    ##

    _defaultLand: dt.EnumPair = None

    def default_land(self):
        if self._defaultLand:
            return self._defaultLand

        with self.db.connect() as conn:
            sel = (
                sa
                .select(self.table(TABLE_PLACE))
                .where(sa.text("data->>'kind' = 'gemarkung'"))
                .limit(1)
            )
            for r in conn.execute(sel):
                p = unserialize(r.data)
                self._defaultLand = p.land
                gws.log.debug(f'ALKIS: defaultLand={vars(self._defaultLand)}')
                return self._defaultLand

    _strasseList: list[dt.Strasse] = []

    def strasse_list(self) -> list[dt.Strasse]:

        if self._strasseList:
            return self._strasseList

        indexlage = self.table(TABLE_INDEXLAGE)
        cols = (
            indexlage.c.gemeinde,
            indexlage.c.gemeindecode,
            indexlage.c.gemarkung,
            indexlage.c.gemarkungcode,
            indexlage.c.strasse,
        )

        self._strasseList = []

        with self.db.connect() as conn:
            for r in conn.execute(sa.select(*cols).group_by(*cols)):
                self._strasseList.append(dt.Strasse(
                    gemeinde=dt.EnumPair(r.gemeindecode, r.gemeinde),
                    gemarkung=dt.EnumPair(r.gemarkungcode, r.gemarkung),
                    name=r.strasse
                ))

        return self._strasseList

    def find_adresse(self, q: dt.AdresseQuery) -> list[dt.Adresse]:
        indexlage = self.table(TABLE_INDEXLAGE)

        qo = q.options or gws.Data()
        sel = self._make_adresse_select(q, qo)

        lage_uids = []
        adresse_map = {}

        with self.db.connect() as conn:
            for r in conn.execute(sel):
                lage_uids.append(r[0])

            if qo.hardLimit and len(lage_uids) > qo.hardLimit:
                raise gws.ResponseTooLargeError(len(lage_uids))

            if qo.offset:
                lage_uids = lage_uids[qo.offset:]
            if qo.limit:
                lage_uids = lage_uids[:qo.limit]

            sel = indexlage.select().where(indexlage.c.lageuid.in_(lage_uids))

            for r in conn.execute(sel):
                r = gws.u.to_dict(r)
                uid = r['lageuid']
                adresse_map[uid] = dt.Adresse(
                    uid=uid,
                    land=dt.EnumPair(r['landcode'], r['land']),
                    regierungsbezirk=dt.EnumPair(r['regierungsbezirkcode'], r['regierungsbezirk']),
                    kreis=dt.EnumPair(r['kreiscode'], r['kreis']),
                    gemeinde=dt.EnumPair(r['gemeindecode'], r['gemeinde']),
                    gemarkung=dt.EnumPair(r['gemarkungcode'], r['gemarkung']),
                    strasse=r['strasse'],
                    hausnummer=r['hausnummer'],
                    x=r['x'],
                    y=r['y'],
                    shape=gws.base.shape.from_xy(r['x'], r['y'], crs=self.crs),
                )

        return gws.u.compact(adresse_map.get(uid) for uid in lage_uids)

    def find_flurstueck(self, q: dt.FlurstueckQuery) -> list[dt.Flurstueck]:
        qo = q.options or gws.Data()
        sel = self._make_flurstueck_select(q, qo)

        fs_uids = []

        with self.db.connect() as conn:
            for r in conn.execute(sel):
                uid = r[0].partition('_')[0]
                if uid not in fs_uids:
                    fs_uids.append(uid)

            if qo.hardLimit and len(fs_uids) > qo.hardLimit:
                raise gws.ResponseTooLargeError(len(fs_uids))

            if qo.offset:
                fs_uids = fs_uids[qo.offset:]
            if qo.limit:
                fs_uids = fs_uids[:qo.limit]

            fs_list = self._load_flurstueck(conn, fs_uids, qo)

        return fs_list

    def count_all(self, qo: dt.FlurstueckQueryOptions) -> int:
        indexfs = self.table(TABLE_INDEXFLURSTUECK)
        sel = sa.select(sa.func.count()).select_from(indexfs)
        if not qo.withHistorySearch:
            sel = sel.where(~indexfs.c.fshistoric)

        with self.db.connect() as conn:
            r = list(conn.execute(sel))
            return r[0][0]

    def iter_all(self, qo: dt.FlurstueckQueryOptions) -> Iterable[dt.Flurstueck]:
        indexfs = self.table(TABLE_INDEXFLURSTUECK)
        sel = sa.select(indexfs.c.fs).with_only_columns(indexfs.c.fs).order_by(indexfs.c.n)
        if not qo.withHistorySearch:
            sel = sel.where(~indexfs.c.fshistoric)

        offset = 0

        # NB the consumer might be slow, close connection on each chunk

        while True:
            with self.db.connect() as conn:
                sel2 = sel.offset(offset).limit(qo.limit)
                fs_uids = [r[0] for r in conn.execute(sel2)]
                if not fs_uids:
                    break
                fs_list = self._load_flurstueck(conn, fs_uids, qo)

            yield from fs_list

            offset += qo.limit

    HAUSNUMMER_NOT_NULL_VALUE = '*'

    def _make_flurstueck_select(self, q: dt.FlurstueckQuery, qo: dt.FlurstueckQueryOptions):

        indexfs = self.table(TABLE_INDEXFLURSTUECK)
        indexbuchungsblatt = self.table(TABLE_INDEXBUCHUNGSBLATT)
        indexgeom = self.table(TABLE_INDEXGEOM)
        indexlage = self.table(TABLE_INDEXLAGE)
        indexperson = self.table(TABLE_INDEXPERSON)

        where = []

        has_buchungsblatt = False
        has_geom = False
        has_lage = False
        has_person = False

        where.extend(self._make_places_where(q, indexfs))

        if q.uids:
            where.append(indexfs.c.fs.in_(q.uids))

        for f in 'flurnummer', 'flurstuecksfolge', 'zaehler', 'nenner':
            val = getattr(q, f, None)
            if val is not None:
                where.append(getattr(indexfs.c, f.lower()) == val)

        if q.flurstueckskennzeichen:
            val = re.sub(r'[^0-9_]', '', q.flurstueckskennzeichen)
            if not val:
                raise gws.BadRequestError(f'invalid flurstueckskennzeichen {q.flurstueckskennzeichen!r}')
            where.append(indexfs.c.flurstueckskennzeichen.like(val + '%'))

        if q.flaecheVon:
            try:
                where.append(indexfs.c.amtlicheflaeche >= float(q.flaecheVon))
            except ValueError:
                raise gws.BadRequestError(f'invalid flaecheVon {q.flaecheVon!r}')

        if q.flaecheBis:
            try:
                where.append(indexfs.c.amtlicheflaeche <= float(q.flaecheBis))
            except ValueError:
                raise gws.BadRequestError(f'invalid flaecheBis {q.flaecheBis!r}')

        if q.buchungsblattkennzeichenList:
            ws = []

            for s in q.buchungsblattkennzeichenList:
                w = text_search_clause(indexbuchungsblatt.c.buchungsblattkennzeichen, s, qo.buchungsblattSearchOptions)
                if w is not None:
                    ws.append(w)
            if ws:
                has_buchungsblatt = True
                where.append(sa.or_(*ws))

        if q.strasse:
            w = text_search_clause(indexlage.c.strasse_t, strasse_key(q.strasse), qo.strasseSearchOptions)
            if w is not None:
                has_lage = True
                where.append(w)

        if q.hausnummer:
            if not has_lage:
                raise gws.BadRequestError(f'hausnummer without strasse')
            if q.hausnummer == self.HAUSNUMMER_NOT_NULL_VALUE:
                where.append(indexlage.c.hausnummer.is_not(None))
            else:
                where.append(indexlage.c.hausnummer == normalize_hausnummer(q.hausnummer))

        if q.personName:
            w = text_search_clause(indexperson.c.name_t, text_key(q.personName), qo.nameSearchOptions)
            if w is not None:
                has_person = True
                where.append(w)

        if q.personVorname:
            if not has_person:
                raise gws.BadRequestError(f'personVorname without personName')
            w = text_search_clause(indexperson.c.vorname_t, text_key(q.personVorname), qo.nameSearchOptions)
            if w is not None:
                where.append(w)

        if q.shape:
            has_geom = True
            where.append(sa.func.st_intersects(
                indexgeom.c.geom,
                sa.cast(q.shape.transformed_to(self.crs).to_ewkb_hex(), sa.geo.Geometry())))

        join = []

        if has_buchungsblatt:
            join.append([indexbuchungsblatt, indexbuchungsblatt.c.fs == indexfs.c.fs])
            if not qo.withHistorySearch:
                where.append(~indexbuchungsblatt.c.fshistoric)
                where.append(~indexbuchungsblatt.c.buchungsblatthistoric)

        if has_geom:
            join.append([indexgeom, indexgeom.c.fs == indexfs.c.fs])
            if not qo.withHistorySearch:
                where.append(~indexgeom.c.fshistoric)

        if has_lage:
            join.append([indexlage, indexlage.c.fs == indexfs.c.fs])
            if not qo.withHistorySearch:
                where.append(~indexlage.c.fshistoric)
                where.append(~indexlage.c.lagehistoric)

        if has_person:
            join.append([indexperson, indexperson.c.fs == indexfs.c.fs])
            if not qo.withHistorySearch:
                where.append(~indexperson.c.fshistoric)
                where.append(~indexperson.c.personhistoric)

        if not qo.withHistorySearch:
            where.append(~indexfs.c.fshistoric)

        sel = sa.select(sa.distinct(indexfs.c.fs))

        for tab, cond in join:
            sel = sel.join(tab, cond)

        sel = sel.where(*where)

        return self._make_sort(sel, qo.sort, indexfs)

    def _make_adresse_select(self, q: dt.AdresseQuery, qo: dt.AdresseQueryOptions):
        indexlage = self.table(TABLE_INDEXLAGE)
        where = []

        where.extend(self._make_places_where(q, indexlage))

        has_strasse = False

        if q.strasse:
            w = text_search_clause(indexlage.c.strasse_t, strasse_key(q.strasse), qo.strasseSearchOptions)
            if w is not None:
                has_strasse = True
                where.append(w)

        if q.hausnummer:
            if not has_strasse:
                raise gws.BadRequestError(f'hausnummer without strasse')
            if q.hausnummer == self.HAUSNUMMER_NOT_NULL_VALUE:
                where.append(indexlage.c.hausnummer.is_not(None))
            else:
                where.append(indexlage.c.hausnummer == normalize_hausnummer(q.hausnummer))

        if q.bisHausnummer:
            if not has_strasse:
                raise gws.BadRequestError(f'hausnummer without strasse')
            where.append(indexlage.c.hausnummer < normalize_hausnummer(q.bisHausnummer))

        if q.hausnummerNotNull:
            if not has_strasse:
                raise gws.BadRequestError(f'hausnummer without strasse')
            where.append(indexlage.c.hausnummer.is_not(None))

        if not qo.withHistorySearch:
            where.append(~indexlage.c.lagehistoric)

        sel = sa.select(sa.distinct(indexlage.c.lageuid))

        sel = sel.where(*where)

        return self._make_sort(sel, qo.sort, indexlage)

    def _make_places_where(self, q: dt.FlurstueckQuery | dt.AdresseQuery, table: sa.Table):
        where = []
        land_code = ''

        for f in 'land', 'regierungsbezirk', 'kreis', 'gemarkung', 'gemeinde':
            val = getattr(q, f, None)
            if val is not None:
                where.append(getattr(table.c, f.lower() + '_t') == text_key(val))

            val = getattr(q, f + 'Code', None)
            if val is not None:
                if f == 'land':
                    land_code = val
                elif f == 'gemarkung' and len(val) <= 4:
                    if not land_code:
                        land = self.default_land()
                        if land:
                            land_code = land.code
                    val = land_code + val

                where.append(getattr(table.c, f.lower() + 'code') == val)

        return where

    def _make_sort(self, sel, sort, table: sa.Table):
        if not sort:
            return sel

        order = []
        for s in sort:
            fn = sa.desc if s.reverse else sa.asc
            order.append(fn(getattr(table.c, s.fieldName)))
        sel = sel.order_by(*order)

        return sel

    def load_flurstueck(self, fs_uids: list[str], qo: dt.FlurstueckQueryOptions) -> list[dt.Flurstueck]:
        with self.db.connect() as conn:
            return self._load_flurstueck(conn, fs_uids, qo)

    def _load_flurstueck(self, conn, fs_uids, qo: dt.FlurstueckQueryOptions):
        with_lage = dt.DisplayTheme.lage in qo.displayThemes
        with_gebaeude = dt.DisplayTheme.gebaeude in qo.displayThemes
        with_nutzung = dt.DisplayTheme.nutzung in qo.displayThemes
        with_festlegung = dt.DisplayTheme.festlegung in qo.displayThemes
        with_bewertung = dt.DisplayTheme.bewertung in qo.displayThemes
        with_buchung = dt.DisplayTheme.buchung in qo.displayThemes
        with_eigentuemer = dt.DisplayTheme.eigentuemer in qo.displayThemes

        tab = self.table(TABLE_FLURSTUECK)
        sel = sa.select(tab).where(tab.c.uid.in_(set(fs_uids)))

        hd = qo.withHistoryDisplay

        fs_list = []

        for r in conn.execute(sel):
            fs = unserialize(r.data)
            fs.geom = r.geom
            fs_list.append(fs)

        fs_list = self._remove_historic(fs_list, hd)
        if not fs_list:
            return []

        fs_map = {fs.uid: fs for fs in fs_list}

        for fs in fs_map.values():
            fs.shape = gws.base.shape.from_wkb_element(fs.geom, default_crs=self.crs)

            fs.lageList = self._remove_historic(fs.lageList, hd) if with_lage else []
            fs.gebaeudeList = self._remove_historic(fs.gebaeudeList, hd) if with_gebaeude else []
            fs.buchungList = self._remove_historic(fs.buchungList, hd) if with_buchung else []

            fs.bewertungList = []
            fs.festlegungList = []
            fs.nutzungList = []

        if with_buchung:
            bb_uids = set(
                bu.buchungsblattUid
                for fs in fs_map.values()
                for bu in fs.buchungList
            )

            tab = self.table(TABLE_BUCHUNGSBLATT)
            sel = sa.select(tab).where(tab.c.uid.in_(bb_uids))
            bb_list = [unserialize(r.data) for r in conn.execute(sel)]
            bb_list = self._remove_historic(bb_list, hd)

            for bb in bb_list:
                bb.buchungsstelleList = self._remove_historic(bb.buchungsstelleList, hd)
                bb.namensnummerList = self._remove_historic(bb.namensnummerList, hd) if with_eigentuemer else []
                for nn in bb.namensnummerList:
                    nn.personList = self._remove_historic(nn.personList, hd)
                    for pe in nn.personList:
                        pe.anschriftList = self._remove_historic(pe.anschriftList, hd)

            bb_map = {bb.uid: bb for bb in bb_list}

            for fs in fs_map.values():
                for bu in fs.buchungList:
                    bu.buchungsblatt = bb_map.get(bu.buchungsblattUid, hd)

        if with_nutzung or with_festlegung or with_bewertung:
            tab = self.table(TABLE_PART)
            sel = sa.select(tab).where(tab.c.fs.in_(list(fs_map)))
            if not qo.withHistorySearch:
                sel.where(~tab.c.parthistoric)
            pa_list = [unserialize(r.data) for r in conn.execute(sel)]
            pa_list = self._remove_historic(pa_list, hd)

            for pa in pa_list:
                fs = fs_map[pa.fs]
                if pa.kind == dt.PART_NUTZUNG and with_nutzung:
                    fs.nutzungList.append(pa)
                if pa.kind == dt.PART_FESTLEGUNG and with_festlegung:
                    fs.festlegungList.append(pa)
                if pa.kind == dt.PART_BEWERTUNG and with_bewertung:
                    fs.bewertungList.append(pa)

        return gws.u.compact(fs_map.get(uid) for uid in fs_uids)

    _historicKeys = [
        'vorgaengerFlurstueckskennzeichen'
    ]

    def _remove_historic(self, objects, with_history_display: bool):
        if with_history_display:
            return objects

        out = []

        for o in objects:
            if o.isHistoric:
                continue

            o.recs = [r for r in o.recs if not r.isHistoric]
            if not o.recs:
                continue

            for r in o.recs:
                for k in self._historicKeys:
                    try:
                        delattr(r, k)
                    except AttributeError:
                        pass

            out.append(o)

        return out


##

def serialize(o: dt.Object, encode_enum_pairs=True) -> dict:
    def encode(r):
        if not r:
            return r

        if isinstance(r, (int, float, str, bool)):
            return r

        if isinstance(r, (datetime.date, datetime.datetime)):
            return f'{r.day:02}.{r.month:02}.{r.year:04}'

        if isinstance(r, list):
            return [encode(e) for e in r]

        if isinstance(r, dt.EnumPair):
            if encode_enum_pairs:
                return f'${r.code}${r.text}'
            return vars(r)

        if isinstance(r, dt.Object):
            return {k: encode(v) for k, v in sorted(vars(r).items())}

        return str(r)

    return encode(o)


def unserialize(data: dict):
    def decode(r):
        if not r:
            return r
        if isinstance(r, str):
            if r[0] == '$':
                s = r.split('$')
                return dt.EnumPair(s[1], s[2])
            return r
        if isinstance(r, list):
            return [decode(e) for e in r]
        if isinstance(r, dict):
            d = {k: decode(v) for k, v in r.items()}
            return dt.Object(**d)
        return r

    return decode(data)


##


def text_key(s):
    """Normalize a text string for full-text search."""

    if s is None:
        return ''

    s = _text_umlauts(str(s).strip().lower())
    return _text_nopunct(s)


def strasse_key(s):
    """Normalize a steet name for full-text search."""

    if s is None:
        return ''

    s = _text_umlauts(str(s).strip().lower())

    s = re.sub(r'\s?str\.$', '.strasse', s)
    s = re.sub(r'\s?pl\.$', '.platz', s)
    s = re.sub(r'\s?(strasse|allee|damm|gasse|pfad|platz|ring|steig|wall|weg|zeile)$', r'.\1', s)

    return _text_nopunct(s)


def _text_umlauts(s):
    s = s.replace(u'ä', 'ae')
    s = s.replace(u'ë', 'ee')
    s = s.replace(u'ö', 'oe')
    s = s.replace(u'ü', 'ue')
    s = s.replace(u'ß', 'ss')

    return s


def _text_nopunct(s):
    return re.sub(r'\W+', ' ', s)


def normalize_hausnummer(s):
    """Clean up house number formatting."""

    if s is None:
        return ''

    # "12 a" -> "12a"
    s = re.sub(r'\s+', '', s.strip())
    return s


def make_fsnummer(r: dt.FlurstueckRecord):
    """Create a 'fsnummer' for a Flurstueck, which is 'flur-zaeher/nenner (folge)'."""

    v = r.gemarkung.code + ' '

    s = r.flurnummer
    if s:
        v += str(s) + '-'

    v += str(r.zaehler)
    s = r.nenner
    if s:
        v += '/' + str(s)

    s = r.flurstuecksfolge
    if s and str(s) != '00':
        v += ' (' + str(s) + ')'

    return v


# parse a fsnummer in the above format, all parts are optional

_RE_FSNUMMER = r'''(?x)
    ^
    (
        (?P<gemarkungCode> [0-9]+)
        \s+
    )?
    (
        (?P<flurnummer> [0-9]+)
        -
    )?
    (
        (?P<zaehler> [0-9]+)
        (/
            (?P<nenner> \w+)
        )?
    )?
    (
        \s*
        \(
            (?P<flurstuecksfolge> [0-9]+)
        \)
    )?
    $
'''


def parse_fsnummer(s):
    """Parse a Flurstueck fsnummer into parts."""

    m = re.match(_RE_FSNUMMER, s.strip())
    if not m:
        return None
    return gws.u.compact(m.groupdict())


def text_search_clause(column, val, tso: gws.TextSearchOptions):
    # @TODO merge with model_field/text

    if val is None:
        return

    val = str(val).strip()
    if len(val) == 0:
        return

    if not tso:
        return column == val

    if tso.minLength and len(val) < tso.minLength:
        return

    if tso.type == gws.TextSearchType.exact:
        return column == val

    if tso.type == gws.TextSearchType.any:
        val = '%' + _escape_like(val) + '%'
    if tso.type == gws.TextSearchType.begin:
        val = _escape_like(val) + '%'
    if tso.type == gws.TextSearchType.end:
        val = '%' + _escape_like(val)

    if tso.caseSensitive:
        return column.like(val, escape='\\')

    return column.ilike(val, escape='\\')


def _escape_like(s, escape='\\'):
    return (
        s
        .replace(escape, escape + escape)
        .replace('%', escape + '%')
        .replace('_', escape + '_'))
