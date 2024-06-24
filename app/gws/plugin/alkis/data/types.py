import gws
import gws.types as t


class EnumPair:
    def __init__(self, code, text):
        self.code = code
        self.text = text


class Strasse(gws.Data):
    """Strasse (street) Record"""

    name: str
    gemarkung: EnumPair
    gemeinde: EnumPair


class Object:
    uid: str
    isHistoric: bool

    def __init__(self, **kwargs):
        self.isHistoric = False
        self.__dict__.update(kwargs)


def _getattr(self, item):
    if item.startswith('_'):
        raise AttributeError()
    return None


setattr(Object, '__getattr__', _getattr)


class Record(Object):
    anlass: str
    beginnt: str
    endet: str


class Entity(Object):
    recs: list['Record']


class Adresse(Object):
    land: EnumPair
    regierungsbezirk: EnumPair
    kreis: EnumPair
    gemeinde: EnumPair
    gemarkung: EnumPair

    strasse: str
    hausnummer: str

    x: float
    y: float
    shape: gws.IShape


class FlurstueckRecord(Record):
    flurnummer: str
    zaehler: str
    nenner: str
    flurstuecksfolge: str
    flurstueckskennzeichen: str

    land: EnumPair
    regierungsbezirk: EnumPair
    kreis: EnumPair
    gemeinde: EnumPair
    gemarkung: EnumPair

    amtlicheFlaeche: float

    geom: str
    geomFlaeche: float
    x: float
    y: float

    abweichenderRechtszustand: bool
    rechtsbehelfsverfahren: bool
    zeitpunktDerEntstehung: str
    zustaendigeStelle: list[EnumPair]
    zweifelhafterFlurstuecksnachweis: bool
    nachfolgerFlurstueckskennzeichen: list[str]


class BuchungsstelleReference(Object):
    buchungsstelle: 'Buchungsstelle'


class Buchung(Entity):
    recs: list['BuchungsstelleReference']
    buchungsblattUid: str
    buchungsblatt: 'Buchungsblatt'


class Flurstueck(Entity):
    recs: list[FlurstueckRecord]

    flurstueckskennzeichen: str
    fsnummer: str

    buchungList: list['Buchung']
    lageList: list['Lage']

    gebaeudeList: list['Gebaeude']
    gebaeudeAmtlicheFlaeche: float
    gebaeudeGeomFlaeche: float

    nutzungList: list['Part']
    festlegungList: list['Part']
    bewertungList: list['Part']

    geom: t.Any
    x: float
    y: float
    shape: gws.IShape


class BuchungsblattRecord(Record):
    blattart: EnumPair
    buchungsart: str
    buchungsblattbezirk: EnumPair
    buchungsblattkennzeichen: str
    buchungsblattnummerMitBuchstabenerweiterung: str


class Buchungsblatt(Entity):
    recs: list[BuchungsblattRecord]
    buchungsstelleList: list['Buchungsstelle']
    namensnummerList: list['Namensnummer']
    buchungsblattkennzeichen: str


class BuchungsstelleRecord(Record):
    anteil: str
    beschreibungDesSondereigentums: str
    beschreibungDesUmfangsDerBuchung: str
    buchungsart: EnumPair
    buchungstext: str
    laufendeNummer: str


class Buchungsstelle(Entity):
    recs: list[BuchungsstelleRecord]
    buchungsblattUids: list[str]
    buchungsblattkennzeichenList: list[str]
    parentUids: list[str]
    childUids: list[str]
    fsUids: list[str]
    parentkennzeichenList: list[str]
    flurstueckskennzeichenList: list[str]
    laufendeNummer: str


class NamensnummerRecord(Record):
    anteil: str
    artDerRechtsgemeinschaft: EnumPair
    beschriebDerRechtsgemeinschaft: str
    eigentuemerart: EnumPair
    laufendeNummerNachDIN1421: str
    nummer: str
    strichblattnummer: int


class Namensnummer(Entity):
    recs: list[NamensnummerRecord]
    buchungsblattUids: list[str]
    buchungsblattkennzeichenList: list[str]
    personList: list['Person']
    laufendeNummer: str


class PersonRecord(Record):
    akademischerGrad: str
    anrede: str
    geburtsdatum: str
    geburtsname: str
    nachnameOderFirma: str
    vorname: str


class Person(Entity):
    recs: list[PersonRecord]
    anschriftList: list['Anschrift']


class AnschriftRecord(Record):
    hausnummer: str
    ort: str
    plz: str
    strasse: str
    telefon: str


class Anschrift(Entity):
    recs: list[AnschriftRecord]


class LageRecord(Record):
    hausnummer: str
    laufendeNummer: str
    ortsteil: str
    pseudonummer: str
    strasse: str


class Lage(Entity):
    recs: list['LageRecord']
    gebaeudeList: list['Gebaeude']
    x: float
    y: float


class GebaeudeRecord(Record):
    gebaeudekennzeichen: int
    props: list[tuple]
    geom: str
    amtlicheFlaeche: float
    geomFlaeche: float


class Gebaeude(Entity):
    PROP_KEYS = {
        'anzahlDerOberirdischenGeschosse',
        'anzahlDerUnterirdischenGeschosse',
        'baujahr',
        'bauweise',
        'dachart',
        'dachform',
        'dachgeschossausbau',
        'gebaeudefunktion',
        'geschossflaeche',
        'hochhaus',
        'lageZurErdoberflaeche',
        'name',
        'objekthoehe',
        'weitereGebaeudefunktion',
        'umbauterRaum',
        'zustand',
    }
    recs: list[GebaeudeRecord]


PART_NUTZUNG = 1
PART_BEWERTUNG = 2
PART_FESTLEGUNG = 3


class PartRecord(Record):
    props: list[tuple]
    amtlicheFlaeche: float  # corrected
    geomFlaeche: float
    geom: str


class Part(Entity):
    KIND = {
        PART_NUTZUNG: [
            'Tatsächliche Nutzung',
            'tatsaechliche_nutzung'
        ],
        PART_BEWERTUNG: [
            'Bodenschätzung, Bewertung',
            'gesetzliche_festlegungen_gebietseinheiten_kataloge/bodenschaetzung_bewertung'
        ],
        PART_FESTLEGUNG: [
            'Öffentlich-rechtliche und sonstige Festlegungen',
            'gesetzliche_festlegungen_gebietseinheiten_kataloge/oeffentlich_rechtliche_und_sonstige_festlegungen'
        ],
    }

    PROP_KEYS = {
        'abbaugut',
        'ackerzahlOderGruenlandzahl',
        'anzahlDerFahrstreifen',
        'anzahlDerStreckengleise',
        'art',
        'artDerBebauung',
        'artDerFestlegung',
        'artDerGebietsgrenze',
        'artDerVerbandsgemeinde',
        'ausfuehrendeStelle',
        'bahnkategorie',
        'baublockbezeichnung',
        'bedeutung',
        'befestigung',
        'besondereFahrstreifen',
        'besondereFunktion',
        'bezeichnung',
        'bodenart',
        'bodenzahlOderGruenlandgrundzahl',
        'breiteDerFahrbahn',
        'breiteDesGewaessers',
        'breiteDesVerkehrsweges',
        'datumAbgabe',
        'datumAnordnung',
        'datumBesitzeinweisung',
        'datumrechtskraeftig',
        'einwohnerzahl',
        'elektrifizierung',
        'entstehungsartOderKlimastufeWasserverhaeltnisse',
        'fliessrichtung',
        'foerdergut',
        'funktion',
        'gehoertZu',
        'gemeindeflaeche',
        'gewaesserkennziffer',
        'hydrologischesMerkmal',
        'jahreszahl',
        'kennziffer',
        'klassifizierung',
        'kulturart',
        'lagergut',
        'landschaftstyp',
        'markierung',
        'merkmal',
        'name',
        'nummer',
        'nummerDerBahnstrecke',
        'nummerDerSchutzzone',
        'nutzung',
        'oberflaechenmaterial',
        'primaerenergie',
        'rechtszustand',
        'schifffahrtskategorie',
        'sonstigeAngaben',
        'spurweite',
        'tagesabschnittsnummer',
        'tidemerkmal',
        'vegetationsmerkmal',
        'veraenderungOhneRuecksprache',
        'verkehrsbedeutungInneroertlich',
        'verkehrsbedeutungUeberoertlich',
        'verwaltungsgemeinschaft',
        'widmung',
        'zone',
        'zustand',
        'zustandsstufeOderBodenstufe',
        'zweitname',
    }

    recs: list['PartRecord']
    fs: str
    kind: int
    name: EnumPair
    amtlicheFlaeche: float
    geomFlaeche: float
    geom: str


class PlaceKind(gws.Enum):
    land = 'land'
    regierungsbezirk = 'regierungsbezirk'
    kreis = 'kreis'
    gemeinde = 'gemeinde'
    gemarkung = 'gemarkung'
    buchungsblattbezirk = 'buchungsblattbezirk'
    dienststelle = 'dienststelle'


class Place(Record):
    kind: str
    land: EnumPair
    regierungsbezirk: EnumPair
    kreis: EnumPair
    gemeinde: EnumPair
    gemarkung: EnumPair
    buchungsblattbezirk: EnumPair
    dienststelle: EnumPair


##

class DisplayTheme(t.Enum):
    lage = 'lage'
    gebaeude = 'gebaeude'
    nutzung = 'nutzung'
    festlegung = 'festlegung'
    bewertung = 'bewertung'
    buchung = 'buchung'
    eigentuemer = 'eigentuemer'


EigentuemerAccessRequired = ['personName', 'personVorname']

BuchungAccessRequired = ['buchungsblattkennzeichenList']


class FlurstueckQueryOptions(gws.Data):
    strasseSearchOptions: gws.TextSearchOptions
    nameSearchOptions: gws.TextSearchOptions
    buchungsblattSearchOptions: gws.TextSearchOptions

    limit: int
    offset: int
    hardLimit: int
    sort: t.Optional[list[gws.SortOptions]]

    displayThemes: list[DisplayTheme]

    withEigentuemer: bool
    withBuchung: bool
    withHistorySearch: bool
    withHistoryDisplay: bool


class FlurstueckQuery(gws.Data):
    flurnummer: str
    flurstuecksfolge: str
    zaehler: str
    nenner: str
    flurstueckskennzeichen: str

    flaecheBis: float
    flaecheVon: float

    gemarkung: str
    gemarkungCode: str
    gemeinde: str
    gemeindeCode: str
    kreis: str
    kreisCode: str
    land: str
    landCode: str
    regierungsbezirk: str
    regierungsbezirkCode: str

    strasse: str
    hausnummer: str

    buchungsblattkennzeichenList: list[str]

    personName: str
    personVorname: str

    shape: gws.IShape

    uids: list[str]

    options: t.Optional['FlurstueckQueryOptions']


class AdresseQueryOptions(gws.Data):
    strasseSearchOptions: gws.TextSearchOptions

    limit: int
    offset: int
    hardLimit: int
    sort: t.Optional[list[gws.SortOptions]]

    withHistorySearch: bool


class AdresseQuery(gws.Data):
    gemarkung: str
    gemarkungCode: str
    gemeinde: str
    gemeindeCode: str
    kreis: str
    kreisCode: str
    land: str
    landCode: str
    regierungsbezirk: str
    regierungsbezirkCode: str

    strasse: str
    hausnummer: str
    bisHausnummer: str
    hausnummerNotNull: bool

    options: t.Optional['AdresseQueryOptions']


##

class Reader:
    def read_all(self, cls: type, table_name: t.Optional[str] = None, uids: t.Optional[list[str]] = None):
        pass

    def count(self, cls: type, table_name: t.Optional[str] = None) -> int:
        pass

##
