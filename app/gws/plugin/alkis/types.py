import gws
import gws.types as t


class Gemarkung(gws.Data):
    """Gemarkung (Administrative Unit) object"""

    gemarkung: str  #: Gemarkung name
    gemarkungUid: str  #: Gemarkung uid
    gemeinde: str  #: Gemeinde name
    gemeindeUid: str  #: Gemeinde uid


class Strasse(gws.Data):
    """Strasse (street) object"""

    strasse: str  #: name
    gemarkung: str  #: Gemarkung name
    gemarkungUid: str  #: Gemarkung uid
    gemeinde: str  #: Gemeinde name
    gemeindeUid: str  #: Gemeinde uid


class StrasseQueryMode(t.Enum):
    exact = 'exact'  #: exact match (up to denormalization)
    substring = 'substring'  #: substring match
    start = 'start'  #: string start match


class BaseQuery(gws.Data):
    gemarkung: str = ''
    gemarkungUid: str = ''
    gemeinde: str = ''
    gemeindeUid: str = ''
    strasse: str = ''
    strasseMode: StrasseQueryMode = StrasseQueryMode.exact


class FindFlurstueckQuery(BaseQuery):
    withEigentuemer: bool = False
    withBuchung: bool = False

    bblatt: str = ''
    bblattMode: str = ''
    flaecheBis: str = ''
    flaecheVon: str = ''
    flurnummer: str = ''
    flurstuecksfolge: str = ''
    fsUids: t.List[str] = []
    hausnummer: str = ''
    name: str = ''
    nenner: str = ''
    strasse: str = ''
    vnum: str = ''
    vorname: str = ''
    zaehler: str = ''

    shape: t.Optional[gws.IShape] = None
    limit: int = 0


class FindFlurstueckResult(gws.Data):
    features: t.List[gws.IFeature] = []
    total: int = 0


class FindAdresseQuery(BaseQuery):
    bisHausnummer: str = ''
    hausnummer: str = ''
    hausnummerNotNull: t.Optional[bool]
    kreis: str = ''
    kreisUid: str = ''
    land: str = ''
    landUid: str = ''
    regierungsbezirk: str = ''
    regierungsbezirkUid: str = ''
    strasse: str = ''

    limit: int = 0


class FindAdresseResult(gws.Data):
    features: t.List[gws.IFeature] = []
    total: int = 0


class FindStrasseQuery(BaseQuery):
    pass


class FindStrasseResult(gws.Data):
    strassen: t.List[Strasse]


class UiGemarkungListMode(t.Enum):
    none = 'none'  #: do not show the list
    plain = 'plain'  #: only "gemarkung"
    combined = 'combined'  #: "gemarkung (gemeinde)"
    tree = 'tree'  # a tree with level 1 = "gemeinde" and level 2 = "gemarkung"


class UiStrasseListMode(t.Enum):
    plain = 'plain'  #: just "strasse"
    withGemeinde = 'withGemeinde'  #: "strasse" ("gemeinde")
    withGemarkung = 'withGemarkung'  #: "strasse" ("gemarkung")
    withGemeindeIfRepeated = 'withGemeindeIfRepeated'  #: "strasse" ("gemeinde"), when needed for disambiguation
    withGemarkungIfRepeated = 'withGemarkungIfRepeated'  #: "strasse" ("gemarkung"), when needed for disambiguation


class UiBblattSearchMode(t.Enum):
    start = 'start'  #: search from the beginning
    end = 'end'  #: search from the end
    any = 'any'  #: search anywhere
    exact = 'exact'  #: exact search


class UiStrasseSearchMode(t.Enum):
    start = 'start'  #: search from the beginning
    any = 'any'  #: search anywhere


class UiOptions(gws.Data):
    """UI options for Flurstückssuche"""

    useExport: bool = False  #: export function enabled
    useSelect: bool = False  #: select mode enabled
    usePick: bool = False  #: pick mode enabled
    searchSelection: bool = False  #: search in selection enabled
    searchSpatial: bool = False  #: spatial search enabled
    gemarkungListMode: UiGemarkungListMode = UiGemarkungListMode.combined  #: gemarkung list mode
    strasseListMode: UiStrasseListMode = UiStrasseListMode.plain  #: strasse list entry format
    strasseSearchMode: UiStrasseSearchMode = UiStrasseSearchMode.start  #: strasse search mode
    autoSpatialSearch: bool = False  #: activate spatial search after submit
    bblattSearchMode: UiBblattSearchMode = UiBblattSearchMode.any  #: buchungsblatt search mode
