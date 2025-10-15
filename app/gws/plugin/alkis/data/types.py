from typing import Optional, Any

import gws


class EnumPair:
    def __init__(self, code, text):
        self.code = code
        self.text = text


class Strasse(gws.Data):
    """Strasse (street) Record"""

    name: str
    """Name of the street."""
    gemarkung: EnumPair
    """Gemarkung (district) of the street."""
    gemeinde: EnumPair
    """Gemeinde (municipality) of the street."""


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
    shape: gws.Shape


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
    vorgaengerFlurstueckskennzeichen: list[str]


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

    geom: Any
    x: float
    y: float
    shape: gws.Shape


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
    amtlicheFlaeche: float
    gebaeudekennzeichen: int
    geom: str
    geomFlaeche: float
    props: 'GebaeudeProps'


class Gebaeude(Entity):
    recs: list[GebaeudeRecord]


PART_NUTZUNG = 1
PART_BEWERTUNG = 2
PART_FESTLEGUNG = 3


class PartRecord(Record):
    amtlicheFlaeche: float  # corrected
    geom: str
    geomFlaeche: float
    props: 'PartProps'


class Part(Entity):
    KIND = {
        PART_NUTZUNG: [
            'Tatsächliche Nutzung',
            'tatsaechliche_nutzung',
        ],
        PART_BEWERTUNG: [
            'Bodenschätzung, Bewertung',
            'gesetzliche_festlegungen_gebietseinheiten_kataloge/bodenschaetzung_bewertung',
        ],
        PART_FESTLEGUNG: [
            'Öffentlich-rechtliche und sonstige Festlegungen',
            'gesetzliche_festlegungen_gebietseinheiten_kataloge/oeffentlich_rechtliche_und_sonstige_festlegungen',
        ],
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


class GebaeudeProps(Object):
    anzahlDerOberirdischenGeschosse: int
    anzahlDerUnterirdischenGeschosse: int
    art: EnumPair
    bauart: EnumPair
    baujahr: list[int]
    bauweise: EnumPair
    beschaffenheit: list[EnumPair]
    dachart: str
    dachform: EnumPair
    dachgeschossausbau: EnumPair
    durchfahrtshoehe: int
    gebaeudefunktion: EnumPair
    gebaeudekennzeichen: str
    geschossflaeche: float
    grundflaeche: float
    hochhaus: bool
    lageZurErdoberflaeche: EnumPair
    name: list[str]
    objekthoehe: int
    punktkennung: str
    sonstigeEigenschaft: list[str]
    umbauterRaum: float
    weitereGebaeudefunktion: list[EnumPair]
    zustand: EnumPair


class PartProps(Object):
    abbaugut: EnumPair
    ackerzahlOderGruenlandzahl: str
    anzahlDerFahrstreifen: int
    anzahlDerStreckengleise: EnumPair
    art: EnumPair
    artDerBebauung: EnumPair
    artDerFestlegung: EnumPair
    bahnkategorie: list[EnumPair]
    bedeutung: list[EnumPair]
    befestigung: EnumPair
    besondereFahrstreifen: EnumPair
    besondereFunktion: EnumPair
    bodenart: EnumPair
    bodenzahlOderGruenlandgrundzahl: str
    breiteDerFahrbahn: int
    breiteDesGewaessers: int
    breiteDesVerkehrsweges: int
    datumAbgabe: str
    datumAnordnung: str
    datumBesitzeinweisung: str
    datumrechtskraeftig: str
    elektrifizierung: EnumPair
    entstehungsartOderKlimastufeWasserverhaeltnisse: list[EnumPair]
    fahrbahntrennung: EnumPair
    fliessrichtung: bool
    foerdergut: EnumPair
    funktion: EnumPair
    gewaesserkennzahl: str
    gewaesserkennziffer: str
    hydrologischesMerkmal: EnumPair
    identnummer: str
    internationaleBedeutung: EnumPair
    jahreszahl: int
    klassifizierung: EnumPair
    kulturart: EnumPair
    lagergut: EnumPair
    markierung: EnumPair
    merkmal: EnumPair
    nummer: str
    nummerDerSchutzzone: str
    nummerDesSchutzgebietes: str
    oberflaechenmaterial: EnumPair
    primaerenergie: EnumPair
    rechtszustand: EnumPair
    schifffahrtskategorie: EnumPair
    sonstigeAngaben: list[EnumPair]
    spurweite: EnumPair
    tagesabschnittsnummer: str
    tidemerkmal: EnumPair
    vegetationsmerkmal: EnumPair
    veraenderungOhneRuecksprache: bool
    verkehrsbedeutungInneroertlich: EnumPair
    verkehrsbedeutungUeberoertlich: EnumPair
    widmung: EnumPair
    zone: EnumPair
    zustand: EnumPair
    zustandsstufeOderBodenstufe: EnumPair


PROPS = {
    'abbaugut': 'Abbaugut',
    'ackerzahlOderGruenlandzahl': 'Ackerzahl oder Grünlandzahl',
    'anzahlDerFahrstreifen': 'Anzahl der Fahrstreifen',
    'anzahlDerOberirdischenGeschosse': 'Anzahl der oberirdischen Geschosse',
    'anzahlDerStreckengleise': 'Anzahl der Streckengleise',
    'anzahlDerUnterirdischenGeschosse': 'Anzahl der unterirdischen Geschosse',
    'art': 'Art',
    'artDerBebauung': 'Art der Bebauung',
    'artDerFestlegung': 'Art der Festlegung',
    'bahnkategorie': 'Bahnkategorie',
    'bauart': 'Bauart',
    'baujahr': 'Baujahr',
    'bauweise': 'Bauweise',
    'bedeutung': 'Bedeutung',
    'befestigung': 'Befestigung',
    'beschaffenheit': 'Beschaffenheit',
    'besondereFahrstreifen': 'Besondere Fahrstreifen',
    'besondereFunktion': 'Besondere Funktion',
    'bodenart': 'Bodenart',
    'bodenzahlOderGruenlandgrundzahl': 'Bodenzahl oder Grünlandgrundzahl',
    'breiteDerFahrbahn': 'Breite der Fahrbahn',
    'breiteDesGewaessers': 'Breite des Gewässers',
    'breiteDesVerkehrsweges': 'Breite des Verkehrsweges',
    'dachart': 'Dachart',
    'dachform': 'Dachform',
    'dachgeschossausbau': 'Dachgeschossausbau',
    'datumAbgabe': 'Datum-Abgabe',
    'datumAnordnung': 'Datum-Anordnung',
    'datumBesitzeinweisung': 'Datum-Besitzeinweisung',
    'datumrechtskraeftig': 'Datum-rechtskräftig',
    'durchfahrtshoehe': 'Durchfahrtshöhe',
    'elektrifizierung': 'Elektrifizierung',
    'entstehungsartOderKlimastufeWasserverhaeltnisse': 'Entstehungsart oder Klimastufe/Wasserverhältnisse',
    'fahrbahntrennung': 'Fahrbahntrennung',
    'fliessrichtung': 'Fließrichtung',
    'foerdergut': 'Fördergut',
    'funktion': 'Funktion',
    'gebaeudefunktion': 'Gebäudefunktion',
    'gebaeudekennzeichen': 'Gebäudekennzeichen',
    'geschossflaeche': 'Geschossfläche',
    'gewaesserkennzahl': 'Gewässerkennzahl',
    'gewaesserkennziffer': 'Gewässerkennziffer',
    'grundflaeche': 'Grundfläche',
    'hochhaus': 'Hochhaus',
    'hydrologischesMerkmal': 'Hydrologisches Merkmal',
    'identnummer': 'Identnummer',
    'internationaleBedeutung': 'Internationale Bedeutung',
    'jahreszahl': 'Jahreszahl',
    'klassifizierung': 'Klassifizierung',
    'kulturart': 'Kulturart',
    'lageZurErdoberflaeche': 'Lage zur Erdoberfläche',
    'lagergut': 'Lagergut',
    'markierung': 'Markierung',
    'merkmal': 'Merkmal',
    'name': 'Name',
    'nummer': 'Nummer',
    'nummerDerSchutzzone': 'Nummer der Schutzzone',
    'nummerDesSchutzgebietes': 'Nummer des Schutzgebietes',
    'oberflaechenmaterial': 'Oberflächenmaterial',
    'objekthoehe': 'Objekthöhe',
    'primaerenergie': 'Primärenergie',
    'punktkennung': 'Punktkennung',
    'rechtszustand': 'Rechtszustand',
    'schifffahrtskategorie': 'Schifffahrtskategorie',
    'sonstigeAngaben': 'Sonstige Angaben',
    'sonstigeEigenschaft': 'Sonstige Eigenschaft',
    'spurweite': 'Spurweite',
    'tagesabschnittsnummer': 'Tagesabschnittsnummer',
    'tidemerkmal': 'Tidemerkmal',
    'umbauterRaum': 'Umbauter Raum',
    'vegetationsmerkmal': 'Vegetationsmerkmal',
    'veraenderungOhneRuecksprache': 'Veränderung ohne Rücksprache',
    'verkehrsbedeutungInneroertlich': 'Verkehrsbedeutung innerörtlich',
    'verkehrsbedeutungUeberoertlich': 'Verkehrsbedeutung überörtlich',
    'weitereGebaeudefunktion': 'Weitere Gebäudefunktion',
    'widmung': 'Widmung',
    'zone': 'Zone',
    'zustand': 'Zustand',
    'zustandsstufeOderBodenstufe': 'Zustandsstufe oder Bodenstufe',
}

##


class DisplayTheme(gws.Enum):
    """Display themes for ALKIS data."""

    lage = 'lage'
    """Lage" theme, showing the location of the property."""
    gebaeude = 'gebaeude'
    """Gebäude" theme, showing building information."""
    nutzung = 'nutzung'
    """Nutzung" theme, showing land use information."""
    festlegung = 'festlegung'
    """Festlegung" theme, showing legal and other determinations."""
    bewertung = 'bewertung'
    """Bewertung" theme, showing valuation information."""
    buchung = 'buchung'
    """Buchung" theme, showing booking information."""
    eigentuemer = 'eigentuemer'
    """Eigentümer" theme, showing owner information."""


EigentuemerAccessRequired = ['personName', 'personVorname']

BuchungAccessRequired = ['buchungsblattkennzeichenList']


class FlurstueckQueryOptions(gws.Data):
    strasseSearchOptions: gws.TextSearchOptions
    nameSearchOptions: gws.TextSearchOptions
    buchungsblattSearchOptions: gws.TextSearchOptions

    limit: int
    pageSize: int
    offset: int
    sort: Optional[list[gws.SortOptions]]

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

    shape: gws.Shape

    uids: list[str]

    options: Optional['FlurstueckQueryOptions']


class AdresseQueryOptions(gws.Data):
    strasseSearchOptions: gws.TextSearchOptions

    limit: int
    pageSize: int
    offset: int
    sort: Optional[list[gws.SortOptions]]

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

    options: Optional['AdresseQueryOptions']


class IndexStatus(gws.Data):
    """Index status"""

    complete: bool
    basic: bool
    eigentuemer: bool
    buchung: bool
    missing: bool


##


class Reader:
    def read_all(self, cls: type, table_name: Optional[str] = None, uids: Optional[list[str]] = None):
        pass

    def count(self, cls: type, table_name: Optional[str] = None) -> int:
        pass


##
