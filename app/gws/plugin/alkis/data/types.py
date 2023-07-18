import gws
import gws.types as t


class EnumPair:
    def __init__(self, code, text):
        self.code = code
        self.text = text


class BoRef:
    bbUid: str
    bbKennzeichen: str
    bsUid: str
    bsLaufendeNummer: str

    def __init__(self, bb_uid='', bb_kennzeichen='', bs_uid='', bs_laufendenummer=''):
        self.bbUid = bb_uid
        self.bbKennzeichen = bb_kennzeichen
        self.bsUid = bs_uid
        self.bsLaufendeNummer = bs_laufendenummer


class Strasse(gws.Data):
    """Strasse (street) Record"""

    name: str
    gemarkung: EnumPair
    gemeinde: EnumPair


class Object:
    uid: str

    def __init__(self, **kwargs):
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

    area: float
    x: float
    y: float
    geom: str
    shape: gws.IShape

    abweichenderRechtszustand: bool
    rechtsbehelfsverfahren: bool
    zeitpunktDerEntstehung: str
    zustaendigeStelle: list[EnumPair]
    zweifelhafterFlurstuecksnachweis: bool
    nachfolgerFlurstueckskennzeichen: list[str]


class Flurstueck(Object):
    recs: list[FlurstueckRecord]

    flurstueckskennzeichen: str

    buchungsstelleRefs: list['BoRef']
    buchungsblattRefs: list['BoRef']

    buchungsblattList: list['Buchungsblatt']

    lageList: list['Lage']

    gebaeudeList: list['Gebaeude']
    gebaeudeGrundflaeche: float
    gebaeudeArea: float

    nutzungList: list['Part']
    festlegungList: list['Part']
    bewertungList: list['Part']

    istHistorisch: bool


class BuchungsblattRecord(Record):
    blattart: EnumPair
    buchungsart: str
    buchungsblattbezirk: EnumPair
    buchungsblattkennzeichen: str
    buchungsblattnummerMitBuchstabenerweiterung: str


class Buchungsblatt(Object):
    recs: list[BuchungsblattRecord]
    buchungsstelleList: list['Buchungsstelle']
    namensnummerList: list['Namensnummer']
    buchungsblattkennzeichen: str
    istHistorisch: bool


class BuchungsstelleRecord(Record):
    anteil: str
    beschreibungDesSondereigentums: str
    beschreibungDesUmfangsDerBuchung: str
    buchungsart: EnumPair
    buchungstext: str
    laufendeNummer: str


class Buchungsstelle(Object):
    recs: list[BuchungsstelleRecord]
    buchungsblattRefs: list['BoRef']
    parentRefs: list['BoRef']
    childRefs: list['BoRef']
    fsUids: list[str]
    flurstueckskennzeichenList: list[str]
    laufendeNummer: str
    istHistorisch: bool


class NamensnummerRecord(Record):
    anteil: str
    artDerRechtsgemeinschaft: EnumPair
    beschriebDerRechtsgemeinschaft: str
    eigentuemerart: EnumPair
    laufendeNummerNachDIN1421: str
    nummer: str
    strichblattnummer: int


class Namensnummer(Object):
    recs: list[NamensnummerRecord]
    buchungsblattRefs: list['BoRef']
    personList: list['Person']
    laufendeNummer: str


class PersonRecord(Record):
    akademischerGrad: str
    anrede: str
    geburtsdatum: str
    geburtsname: str
    nachnameOderFirma: str
    vorname: str


class Person(Object):
    recs: list[PersonRecord]
    anschriftList: list['Anschrift']


class AnschriftRecord(Record):
    hausnummer: str
    ort: str
    plz: str
    strasse: str
    telefon: str


class Anschrift(Object):
    recs: list[AnschriftRecord]


class LageRecord(Record):
    hausnummer: str
    laufendeNummer: str
    ortsteil: str
    pseudonummer: str
    strasse: str


class Lage(Object):
    recs: list['LageRecord']
    gebaeudeList: list['Gebaeude']


class GebaeudeRecord(Record):
    gebaeudekennzeichen: int
    props: list[tuple]
    geom: str
    grundflaeche: float
    area: float


class Gebaeude(Object):
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
    area: float
    geom: str


class Part(Object):
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
        'gemeindekennzeichen',
        'gewaesserkennziffer',
        'hydrologischesMerkmal',
        'inGemarkung',
        'jahreszahl',
        'kennziffer',
        'klassifizierung',
        'kreis',
        'kulturart',
        'lagergut',
        'land',
        'landschaftstyp',
        'markierung',
        'merkmal',
        'name',
        'name',
        'nationalstaat',
        'nummer',
        'nummerDerBahnstrecke',
        'nummerDerSchutzzone',
        'nutzung',
        'oberflaechenmaterial',
        'primaerenergie',
        'rechtszustand',
        'regierungsbezirk',
        'schifffahrtskategorie',
        'sonstigeAngaben',
        'spurweite',
        'strassenschluessel',
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
    area: float
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

BuchungAccessRequired = ['buchungsblattkennzeichen']


class FlurstueckSearchQuery(gws.Data):
    flurnummer: str
    flurstuecksfolge: str
    zaehler: str
    nenner: str
    flurstueckskennzeichen: str

    flaecheBis: float
    flaecheVon: float

    gemarkungCode: str
    gemeindeCode: str

    strasse: str
    hausnummer: str

    buchungsblattkennzeichenList: list[str]

    personName: str
    personVorname: str

    shape: gws.IShape

    uids: list[str]


class FlurstueckSearchOptions(gws.Data):
    strasseSearchOptions: gws.TextSearchOptions
    nameSearchOptions: gws.TextSearchOptions
    buchungsblattSearchOptions: gws.TextSearchOptions
    limit: int
    displayThemes: list[DisplayTheme]
    withHistorySearch: bool
    withHistoryDisplay: bool


##

class Reader:
    def read_all(self, cls: type, table_name: t.Optional[str] = None, uids: t.Optional[list[str]] = None):
        pass

    def count(self, cls: type, table_name: t.Optional[str] = None) -> int:
        pass

##
