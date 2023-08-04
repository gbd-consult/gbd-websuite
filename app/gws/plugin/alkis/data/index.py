import re
import datetime

from sqlalchemy.dialects.postgresql import JSONB

import gws
import gws.base.shape
import gws.base.database
import gws.gis.crs
import gws.plugin.postgres.provider
import gws.lib.sa as sa
from gws.lib.console import ProgressIndicator

import gws.types as t

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
    VERSION = '8'

    TABLE_IDS = [
        TABLE_PLACE,
        TABLE_FLURSTUECK,
        TABLE_BUCHUNGSBLATT,
        TABLE_LAGE,
        TABLE_PART,
        TABLE_INDEXFLURSTUECK,
        TABLE_INDEXLAGE,
        TABLE_INDEXBUCHUNGSBLATT,
        TABLE_INDEXPERSON,
        TABLE_INDEXGEOM,
    ]

    provider: gws.plugin.postgres.provider.Object
    crs: gws.ICrs
    schema: str
    excludeGemarkung: set[str]

    saMeta: sa.MetaData
    tables: dict[str, sa.Table]

    columnDct = {}

    def configure(self):
        self.provider = gws.base.database.provider.get_for(self, ext_type='postgres')
        self.crs = gws.gis.crs.get(self.cfg('crs'))
        self.schema = self.cfg('schema', default='public')
        self.excludeGemarkung = set(self.cfg('excludeGemarkung', default=[]))
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

    def connect(self):
        return self.provider.engine().connect()

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
        with self.connect() as conn:
            return self._table_size(conn, table_id)

    def _table_size(self, conn, table_id):
        tab = self.table(table_id)
        sel = sa.text(f'SELECT COUNT(*) FROM {self.schema}.{tab.name}')
        try:
            for r in conn.execute(sel):
                return r[0]
        except sa.exc.ProgrammingError:
            return 0

    def has_table(self, table_id: str) -> bool:
        return self.table_size(table_id) > 0

    def exists(self):
        with self.connect() as conn:
            for table_id in self.TABLE_IDS:
                size = self._table_size(conn, table_id)
                gws.log.debug(f'{table_id=} {size=}')
                if size == 0:
                    return False

        return True

    def drop_table(self, table_id: str):
        tab = self.table(table_id)
        gws.log.debug(f'drop {tab.name!r}')
        self.saMeta.drop_all(self.provider.engine(), tables=[tab])

    def drop(self):
        tabs = [self.table(table_id) for table_id in self.TABLE_IDS]
        self.saMeta.drop_all(self.provider.engine(), tables=tabs)

    INSERT_SIZE = 5000

    def create_table(
            self,
            table_id: str,
            values: list[dict],
            progress: t.Optional[ProgressIndicator] = None
    ):
        tab = self.table(table_id)
        self.saMeta.create_all(self.provider.engine(), tables=[tab])

        with self.connect() as conn:
            for i in range(0, len(values), self.INSERT_SIZE):
                vals = values[i:i + self.INSERT_SIZE]
                conn.execute(sa.insert(tab).values(vals))
                conn.commit()
                if progress:
                    progress.update(len(vals))

    ##

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

        with self.connect() as conn:
            for r in conn.execute(sa.select(*cols).group_by(*cols)).mappings().all():
                self._strasseList.append(dt.Strasse(
                    gemeinde=dt.EnumPair(r.gemeindecode, r.gemeinde),
                    gemarkung=dt.EnumPair(r.gemarkungcode, r.gemarkung),
                    name=r.strasse
                ))

        return self._strasseList

    def find_flurstueck_uids(self, q: dt.FlurstueckSearchQuery, qo: dt.FlurstueckSearchOptions) -> list[str]:
        indexfs = self.table(TABLE_INDEXFLURSTUECK)

        join, where = self._find_prepare(q, qo)

        sel = sa.select(sa.distinct(indexfs.c.fs))
        for tab, cond in join:
            sel = sel.join(tab, cond)
        sel = sel.where(*where)

        if qo.limit:
            sel = sel.limit(qo.limit)

        fs_uids = set()
        with self.connect() as conn:
            for r in conn.execute(sel):
                fs_uids.add(r[0].partition('_')[0])

        return list(fs_uids)

    def _find_prepare(self, q: dt.FlurstueckSearchQuery, qo: dt.FlurstueckSearchOptions):

        indexfs = self.table(TABLE_INDEXFLURSTUECK)
        indexlage = self.table(TABLE_INDEXLAGE)
        indexperson = self.table(TABLE_INDEXPERSON)
        indexgeom = self.table(TABLE_INDEXGEOM)

        where = []

        has_lage = False
        has_person = False
        has_geom = False

        if q.uids:
            where.append(indexfs.c.fs.in_(q.uids))

        for name in 'flurnummer', 'flurstuecksfolge', 'zaehler', 'nenner':
            val = getattr(q, name, None)
            if val is not None:
                where.append(getattr(indexfs.c, name) == val)

        if q.flurstueckskennzeichen:
            val = re.sub(r'[^0-9_]', '', q.flurstueckskennzeichen)
            if not val:
                raise gws.BadRequestError(f'invalid flurstueckskennzeichen {q.flurstueckskennzeichen!r}')
            where.append(indexfs.c.flurstueckskennzeichen.like(val + '%'))

        if q.gemarkungCode:
            where.append(indexfs.c.gemarkungcode == q.gemarkungCode)
        if q.gemeindeCode:
            where.append(indexfs.c.gemeindecode == q.gemeindeCode)

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
                w = text_search_clause(indexfs.c.buchungsblattkennzeichen, s, qo.buchungsblattSearchOptions)
                if w:
                    ws.append(w)
            if ws:
                where.append(sa.or_(*ws))

        if q.strasse:
            w = text_search_clause(indexlage.c.strasse_t, strasse_key(q.strasse), qo.strasseSearchOptions)
            if w is not None:
                has_lage = True
                where.append(w)

        if q.hausnummer:
            if not has_lage:
                raise gws.BadRequestError(f'hausnummer without strasse')
            if q.hausnummer == '*':
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

        if has_geom:
            join.append([indexgeom, indexgeom.c.fs == indexfs.c.fs])
            if not qo.withHistorySearch:
                where.append(~indexgeom.c.fshistoric)

        if not qo.withHistorySearch:
            where.append(~indexfs.c.fshistoric)

        return join, where

    def load_flurstueck(self, fs_uids: list[str], qo: dt.FlurstueckSearchOptions) -> list[dt.Flurstueck]:
        with self.connect() as conn:
            return self._load_flurstueck(conn, fs_uids, qo)

    def _load_flurstueck(self, conn, fs_uids, qo: dt.FlurstueckSearchOptions):

        def _check_history(objects):
            if qo.withHistoryDisplay:
                return objects
            res = []
            for o in objects:
                if o.isHistoric:
                    continue
                o.recs = [r for r in o.recs if not r.isHistoric]
                if not o.recs:
                    continue
                res.append(o)
            return res

        with_lage = dt.DisplayTheme.lage in qo.displayThemes
        with_gebaeude = dt.DisplayTheme.gebaeude in qo.displayThemes
        with_nutzung = dt.DisplayTheme.nutzung in qo.displayThemes
        with_festlegung = dt.DisplayTheme.festlegung in qo.displayThemes
        with_bewertung = dt.DisplayTheme.bewertung in qo.displayThemes
        with_buchung = dt.DisplayTheme.buchung in qo.displayThemes
        with_eigentuemer = dt.DisplayTheme.eigentuemer in qo.displayThemes

        tab = self.table(TABLE_FLURSTUECK)
        sel = sa.select(tab).where(tab.c.uid.in_(set(fs_uids)))

        fs_list = []

        for r in conn.execute(sel):
            fs = unserialize(r.data)
            fs.geom = r.geom
            fs_list.append(fs)

        fs_list = _check_history(fs_list)
        if not fs_list:
            return []

        fs_map = {fs.uid: fs for fs in fs_list}

        for fs in fs_map.values():
            fs.shape = gws.base.shape.from_wkb_element(fs.geom, default_crs=self.crs)

            fs.lageList = _check_history(fs.lageList) if with_lage else []
            fs.gebaeudeList = _check_history(fs.gebaeudeList) if with_gebaeude else []
            fs.buchungList = _check_history(fs.buchungList) if with_buchung else []

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
            bb_list = _check_history(bb_list)

            for bb in bb_list:
                bb.buchungsstelleList = _check_history(bb.buchungsstelleList)
                bb.namensnummerList = _check_history(bb.namensnummerList) if with_eigentuemer else []
                for nn in bb.namensnummerList:
                    nn.personList = _check_history(nn.personList)
                    for pe in nn.personList:
                        pe.anschriftList = _check_history(pe.anschriftList)

            bb_map = {bb.uid: bb for bb in bb_list}

            for fs in fs_map.values():
                for bu in fs.buchungList:
                    bu.buchungsblatt = bb_map.get(bu.buchungsblattUid)

        if with_nutzung or with_festlegung or with_bewertung:
            tab = self.table(TABLE_PART)
            sel = sa.select(tab).where(tab.c.fs.in_(list(fs_map)))
            if not qo.withHistorySearch:
                sel.where(~tab.c.parthistoric)
            pa_list = [unserialize(r.data) for r in conn.execute(sel)]
            pa_list = _check_history(pa_list)

            for pa in pa_list:
                fs = fs_map[pa.fs]
                if pa.kind == dt.PART_NUTZUNG and with_nutzung:
                    fs.nutzungList.append(pa)
                if pa.kind == dt.PART_FESTLEGUNG and with_festlegung:
                    fs.festlegungList.append(pa)
                if pa.kind == dt.PART_BEWERTUNG and with_bewertung:
                    fs.bewertungList.append(pa)

        return gws.compact(fs_map.get(uid) for uid in fs_uids)


##

def serialize(o: dt.Object) -> dict:
    def encode(r):
        if not r:
            return r

        if isinstance(r, (int, float, str, bool)):
            return r

        if isinstance(r, (datetime.date, datetime.datetime)):
            # return str(r)
            return f'{r.day:02}.{r.month:02}.{r.year:04}'

        if isinstance(r, list):
            return [encode(e) for e in r]

        if isinstance(r, dt.EnumPair):
            return f'${r.code}${r.text}'

        if isinstance(r, dt.Object):
            return {k: encode(v) for k, v in vars(r).items()}

        raise ValueError(f'unserializable object type: {r!r}')

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

    s = str(s).strip().lower()

    s = s.replace(u'ä', 'ae')
    s = s.replace(u'ë', 'ee')
    s = s.replace(u'ö', 'oe')
    s = s.replace(u'ü', 'ue')
    s = s.replace(u'ß', 'ss')

    s = re.sub(r'\W+', ' ', s)
    return s.strip()


def strasse_key(s):
    """Normalize a steet name for full-text search."""

    s = text_key(s)

    s = re.sub(r'\s?str\.$', '.strasse', s)
    s = re.sub(r'\s?pl\.$', '.platz', s)
    s = re.sub(r'\s?(strasse|allee|damm|gasse|pfad|platz|ring|steig|wall|weg|zeile)$', r'.\1', s)

    s = s.replace(' ', '.')
    return s


def normalize_hausnummer(s):
    """Clean up house number formatting."""

    if s is None:
        return ''

    # "12 a" -> "12a"
    s = re.sub(r'\s+', '', s.strip())
    return s


def fs_vollnummer(fs: dt.FlurstueckRecord):
    """Create a 'vollNummer' for a Flurstueck, which is 'flur-zaeher/nenner (folge)'."""

    v = fs.gemarkung.code + ' '

    s = fs.flurnummer
    if s:
        v += str(s) + '-'

    v += fs.zaehler
    s = fs.nenner
    if s:
        v += '/' + str(s)

    s = fs.flurstuecksfolge
    if s and str(s) != '00':
        v += ' (' + str(s) + ')'

    return v


# parse a vollnummer in the above format, all parts are optional

_RE_VOLLNUMMER = r'''(?x)
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


def fs_parse_vollnummer(s):
    """Parse the Flurstueck vollNummer into parts."""

    m = re.match(_RE_VOLLNUMMER, s.strip())
    if not m:
        return None
    return gws.compact(m.groupdict())


def text_search_clause(column, val, tso: gws.TextSearchOptions):
    # @TODO merge with model_field/text

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
