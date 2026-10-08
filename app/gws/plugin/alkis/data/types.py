"""Data model of the ALKIS index: entities, records, queries and the source reader interface."""

from typing import Optional, Any

import gws


class EnumPair:
    """A code list value with its code and display text."""

    def __init__(self, code, text):
        """Create a code list value.

        Args:
            code: Value code.
            text: Display text.
        """

        self.code = code
        self.text = text


class Strasse(gws.Data):
    """A street name in a Gemeinde and Gemarkung."""

    name: str
    """Street name."""
    gemarkung: EnumPair
    """Gemarkung (cadastral district) of the street."""
    gemeinde: EnumPair
    """Gemeinde (municipality) of the street."""


class Object:
    """Base class for ALKIS data objects.

    Keyword arguments passed to the constructor become attributes.
    Reading a missing attribute whose name does not start with ``_``
    returns ``None`` instead of raising ``AttributeError``.
    """

    uid: str
    """Unique id. For ALKIS objects, this is the ALKIS identifier."""
    isHistoric: bool
    """Whether the object is historic."""

    def __init__(self, **kwargs):
        """Create an object.

        Args:
            **kwargs: Attribute values.
        """

        self.isHistoric = False
        self.__dict__.update(kwargs)


def _getattr(self, item):
    if item.startswith('_'):
        raise AttributeError()
    return None


setattr(Object, '__getattr__', _getattr)


class Record(Object):
    """A version of an ALKIS object in its life span.

    A record is historic when its life span has ended.
    """

    anlass: str
    """Reason (Anlass) for the change that created this record."""
    beginnt: str
    """Start of the record life span."""
    endet: str
    """End of the record life span, ``None`` if the record is current."""


class Entity(Object):
    """An ALKIS object with a list of records, one per version.

    Records are sorted by start date, so the last record is the most recent one.
    An entity is historic when all of its records are historic.
    """

    recs: list['Record']
    """Records, oldest first."""


class Adresse(Object):
    """An address (Lagebezeichnung) found by the address search."""

    land: EnumPair
    """Land (federal state)."""
    regierungsbezirk: EnumPair
    """Regierungsbezirk (administrative region)."""
    kreis: EnumPair
    """Kreis (district)."""
    gemeinde: EnumPair
    """Gemeinde (municipality)."""
    gemarkung: EnumPair
    """Gemarkung (cadastral district)."""

    strasse: str
    """Street name."""
    hausnummer: str
    """House number."""

    x: float
    """X coordinate of the address location."""
    y: float
    """Y coordinate of the address location."""
    shape: gws.Shape
    """Point shape of the address location."""


class FlurstueckRecord(Record):
    """A version of a Flurstueck (land parcel)."""

    flurnummer: str
    """Flur number."""
    zaehler: str
    """Numerator of the Flurstueck number."""
    nenner: str
    """Denominator of the Flurstueck number."""
    flurstuecksfolge: str
    """Flurstueck sequence number."""
    flurstueckskennzeichen: str
    """Flurstueck identifier (Flurstueckskennzeichen)."""

    land: EnumPair
    """Land (federal state)."""
    regierungsbezirk: EnumPair
    """Regierungsbezirk (administrative region)."""
    kreis: EnumPair
    """Kreis (district)."""
    gemeinde: EnumPair
    """Gemeinde (municipality)."""
    gemarkung: EnumPair
    """Gemarkung (cadastral district)."""

    amtlicheFlaeche: float
    """Official area in square meters."""

    geom: str
    """Geometry as a hex encoded WKB string."""
    geomFlaeche: float
    """Area computed from the geometry."""
    x: float
    """X coordinate of the geometry centroid."""
    y: float
    """Y coordinate of the geometry centroid."""

    abweichenderRechtszustand: bool
    """Whether a different legal state exists (abweichender Rechtszustand)."""
    rechtsbehelfsverfahren: bool
    """Whether a legal remedy procedure is pending (Rechtsbehelfsverfahren)."""
    zeitpunktDerEntstehung: str
    """Time of creation of the Flurstueck."""
    zustaendigeStelle: list[EnumPair]
    """Responsible authorities (Dienststellen)."""
    zweifelhafterFlurstuecksnachweis: bool
    """Whether the Flurstueck record is doubtful (zweifelhafter Flurstuecksnachweis)."""
    nachfolgerFlurstueckskennzeichen: list[str]
    """Identifiers of successor Flurstuecke."""
    vorgaengerFlurstueckskennzeichen: list[str]
    """Identifiers of predecessor Flurstuecke, computed by the indexer from the successor lists."""


class BuchungsstelleReference(Object):
    """A reference from a Buchung to a Buchungsstelle.

    A Buchungsstelle referred to by a historic Flurstueck record is not
    necessarily historic itself, so the reference has its own historic state,
    which follows the Flurstueck record.
    """

    buchungsstelle: 'Buchungsstelle'
    """Referenced Buchungsstelle."""


class Buchung(Entity):
    """Buchungsstellen of a Flurstueck that belong to the same Buchungsblatt."""

    recs: list['BuchungsstelleReference']
    """References to the Buchungsstellen."""
    buchungsblattUid: str
    """Uid of the Buchungsblatt."""
    buchungsblatt: 'Buchungsblatt'
    """The Buchungsblatt, attached when a Flurstueck is loaded with land register data."""


class Flurstueck(Entity):
    """A Flurstueck (land parcel) with its related data."""

    recs: list[FlurstueckRecord]
    """Flurstueck records, oldest first."""

    flurstueckskennzeichen: str
    """Flurstueck identifier from the most recent record."""
    fsnummer: str
    """Display number from the most recent record, see ``index.make_fsnummer``."""

    buchungList: list['Buchung']
    """Land register data, grouped by Buchungsblatt."""
    lageList: list['Lage']
    """Location designations."""

    gebaeudeList: list['Gebaeude']
    """Buildings, found through the location designations."""
    gebaeudeAmtlicheFlaeche: float
    """Total official area of the current buildings."""
    gebaeudeGeomFlaeche: float
    """Total geometric area of the current buildings."""

    nutzungList: list['Part']
    """Land use (Nutzung) parts."""
    festlegungList: list['Part']
    """Legal designation (Festlegung) parts."""
    bewertungList: list['Part']
    """Soil valuation (Bewertung) parts."""

    geom: Any
    """Geometry as read from the index table."""
    x: float
    """X coordinate of the centroid of the most recent geometry."""
    y: float
    """Y coordinate of the centroid of the most recent geometry."""
    shape: gws.Shape
    """Shape of the Flurstueck, created when it is loaded from the index."""


class BuchungsblattRecord(Record):
    """A version of a Buchungsblatt (land register sheet)."""

    blattart: EnumPair
    """Kind of the sheet (Blattart)."""
    buchungsart: str
    """Kind of the entry (Buchungsart)."""
    buchungsblattbezirk: EnumPair
    """Buchungsblattbezirk (land register district)."""
    buchungsblattkennzeichen: str
    """Sheet identifier (Buchungsblattkennzeichen)."""
    buchungsblattnummerMitBuchstabenerweiterung: str
    """Sheet number with an optional letter suffix."""


class Buchungsblatt(Entity):
    """A Buchungsblatt (land register sheet)."""

    recs: list[BuchungsblattRecord]
    """Buchungsblatt records, oldest first."""
    buchungsstelleList: list['Buchungsstelle']
    """Buchungsstellen in this sheet."""
    namensnummerList: list['Namensnummer']
    """Owner entries in this sheet."""
    buchungsblattkennzeichen: str
    """Sheet identifier from the most recent record."""


class BuchungsstelleRecord(Record):
    """A version of a Buchungsstelle (land register entry)."""

    anteil: str
    """Share as a fraction string, e.g. ``1/2``."""
    beschreibungDesSondereigentums: str
    """Description of the separate ownership (Sondereigentum)."""
    beschreibungDesUmfangsDerBuchung: str
    """Description of the extent of the entry."""
    buchungsart: EnumPair
    """Kind of the entry (Buchungsart)."""
    buchungstext: str
    """Entry text."""
    laufendeNummer: str
    """Sequence number in the sheet."""


class Buchungsstelle(Entity):
    """A Buchungsstelle (land register entry).

    Buchungsstellen can refer to parent Buchungsstellen through the ``an``
    and ``zu`` relations in ALKIS.
    """

    recs: list[BuchungsstelleRecord]
    """Buchungsstelle records, oldest first."""
    buchungsblattUids: list[str]
    """Uids of the sheets this entry belongs to."""
    buchungsblattkennzeichenList: list[str]
    """Identifiers of the sheets this entry belongs to."""
    parentUids: list[str]
    """Uids of the parent Buchungsstellen."""
    childUids: list[str]
    """Uids of the child Buchungsstellen."""
    fsUids: list[str]
    """Uids of the Flurstuecke booked on this entry."""
    parentkennzeichenList: list[str]
    """Identifiers of the parent entries, as ``<buchungsblattkennzeichen>.<laufendeNummer>``."""
    flurstueckskennzeichenList: list[str]
    """Identifiers of the Flurstuecke booked on this entry."""
    laufendeNummer: str
    """Sequence number from the most recent record."""


class NamensnummerRecord(Record):
    """A version of a Namensnummer (owner entry in a land register sheet)."""

    anteil: str
    """Share as a fraction string, e.g. ``1/2``."""
    artDerRechtsgemeinschaft: EnumPair
    """Kind of the joint ownership (Rechtsgemeinschaft)."""
    beschriebDerRechtsgemeinschaft: str
    """Description of the joint ownership."""
    eigentuemerart: EnumPair
    """Kind of the owner."""
    laufendeNummerNachDIN1421: str
    """Sequence number according to DIN 1421."""
    nummer: str
    """Number."""
    strichblattnummer: int
    """Strichblatt number."""


class Namensnummer(Entity):
    """A Namensnummer (owner entry in a land register sheet)."""

    recs: list[NamensnummerRecord]
    """Namensnummer records, oldest first."""
    buchungsblattUids: list[str]
    """Uids of the sheets this entry belongs to."""
    buchungsblattkennzeichenList: list[str]
    """Identifiers of the sheets this entry belongs to."""
    personList: list['Person']
    """Persons named by this entry."""
    laufendeNummer: str
    """Sequence number according to DIN 1421 from the most recent record."""


class PersonRecord(Record):
    """A version of a Person (owner)."""

    akademischerGrad: str
    """Academic title."""
    anrede: str
    """Form of address."""
    geburtsdatum: str
    """Date of birth."""
    geburtsname: str
    """Birth name."""
    nachnameOderFirma: str
    """Last name or company name."""
    vorname: str
    """First name."""


class Person(Entity):
    """A Person (owner), either a natural person or a company."""

    recs: list[PersonRecord]
    """Person records, oldest first."""
    anschriftList: list['Anschrift']
    """Postal addresses."""


class AnschriftRecord(Record):
    """A version of an Anschrift (postal address of a person)."""

    hausnummer: str
    """House number."""
    ort: str
    """City."""
    plz: str
    """Postal code."""
    strasse: str
    """Street name."""
    telefon: str
    """Phone number, the first one if there are several."""


class Anschrift(Entity):
    """An Anschrift (postal address of a person)."""

    recs: list[AnschriftRecord]
    """Anschrift records, oldest first."""


class LageRecord(Record):
    """A version of a Lage (location designation of a Flurstueck)."""

    hausnummer: str
    """Normalized house number, see ``index.normalize_hausnummer``."""
    laufendeNummer: str
    """Sequence number."""
    ortsteil: str
    """Locality."""
    pseudonummer: str
    """Pseudo number."""
    strasse: str
    """Street name."""


class Lage(Entity):
    """A Lage (location designation) with or without a house number."""

    recs: list['LageRecord']
    """Lage records, oldest first."""
    gebaeudeList: list['Gebaeude']
    """Buildings that refer to this location."""
    x: float
    """X coordinate of the house number label, if any."""
    y: float
    """Y coordinate of the house number label, if any."""


class GebaeudeRecord(Record):
    """A version of a Gebaeude (building)."""

    amtlicheFlaeche: float
    """Official area, taken from the building ground area."""
    gebaeudekennzeichen: int
    """Building identifier."""
    geom: str
    """Geometry as a hex encoded WKB string. Not stored in the index."""
    geomFlaeche: float
    """Area computed from the geometry."""
    props: 'GebaeudeProps'
    """Descriptive properties."""


class Gebaeude(Entity):
    """A Gebaeude (building)."""

    recs: list[GebaeudeRecord]
    """Gebaeude records, oldest first."""


PART_NUTZUNG = 1
PART_BEWERTUNG = 2
PART_FESTLEGUNG = 3


class PartRecord(Record):
    """A version of a Part, limited to one Flurstueck."""

    amtlicheFlaeche: float  # corrected
    """Area of the part, scaled by the ratio of official to geometric area of the Flurstueck."""
    geom: str
    """Geometry as a hex encoded WKB string."""
    geomFlaeche: float
    """Area of the intersection geometry."""
    props: 'PartProps'
    """Descriptive properties."""


class Part(Entity):
    """A part of a Flurstueck covered by a Nutzung, Festlegung or Bewertung object.

    Parts are computed by intersecting the geometry of the source object with
    the Flurstueck geometry. There is one part per source object and Flurstueck.
    """

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
    """Part kinds, mapped to their title and the GeoInfoDok category key of the source objects."""

    recs: list['PartRecord']
    """Part records, oldest first."""
    fs: str
    """Uid of the Flurstueck."""
    kind: int
    """Kind of the part, one of ``PART_NUTZUNG``, ``PART_BEWERTUNG`` or ``PART_FESTLEGUNG``."""
    name: EnumPair
    """Object type of the source object, as GeoInfoDok code and title."""
    amtlicheFlaeche: float
    """Corrected area of the last computed intersection."""
    geomFlaeche: float
    """Geometric area of the last computed intersection."""
    geom: str
    """Last computed intersection geometry as a hex encoded WKB string."""


class PlaceKind(gws.Enum):
    """Kind of a place (administrative unit)."""

    land = 'land'
    """Land (federal state)."""
    regierungsbezirk = 'regierungsbezirk'
    """Regierungsbezirk (administrative region)."""
    kreis = 'kreis'
    """Kreis (district)."""
    gemeinde = 'gemeinde'
    """Gemeinde (municipality)."""
    gemarkung = 'gemarkung'
    """Gemarkung (cadastral district)."""
    buchungsblattbezirk = 'buchungsblattbezirk'
    """Buchungsblattbezirk (land register district)."""
    dienststelle = 'dienststelle'
    """Dienststelle (authority)."""


class Place(Record):
    """A place (administrative unit) and the units it belongs to.

    The attribute named by ``kind`` holds the place itself, the other
    attributes hold the higher level units, if known.
    """

    kind: str
    """Kind of the place, a ``PlaceKind`` value."""
    land: EnumPair
    """Land (federal state)."""
    regierungsbezirk: EnumPair
    """Regierungsbezirk (administrative region)."""
    kreis: EnumPair
    """Kreis (district)."""
    gemeinde: EnumPair
    """Gemeinde (municipality)."""
    gemarkung: EnumPair
    """Gemarkung (cadastral district)."""
    buchungsblattbezirk: EnumPair
    """Buchungsblattbezirk (land register district)."""
    dienststelle: EnumPair
    """Dienststelle (authority)."""


##


class GebaeudeProps(Object):
    """Descriptive properties of a building, as defined in GeoInfoDok."""

    anzahlDerOberirdischenGeschosse: int
    """Anzahl der oberirdischen Geschosse."""
    anzahlDerUnterirdischenGeschosse: int
    """Anzahl der unterirdischen Geschosse."""
    art: EnumPair
    """Art."""
    bauart: EnumPair
    """Bauart."""
    baujahr: list[int]
    """Baujahr."""
    bauweise: EnumPair
    """Bauweise."""
    beschaffenheit: list[EnumPair]
    """Beschaffenheit."""
    dachart: str
    """Dachart."""
    dachform: EnumPair
    """Dachform."""
    dachgeschossausbau: EnumPair
    """Dachgeschossausbau."""
    durchfahrtshoehe: int
    """Durchfahrtshoehe."""
    gebaeudefunktion: EnumPair
    """Gebaeudefunktion."""
    gebaeudekennzeichen: str
    """Gebaeudekennzeichen."""
    geschossflaeche: float
    """Geschossflaeche."""
    grundflaeche: float
    """Grundflaeche."""
    hochhaus: bool
    """Hochhaus."""
    lageZurErdoberflaeche: EnumPair
    """Lage zur Erdoberflaeche."""
    name: list[str]
    """Name."""
    objekthoehe: int
    """Objekthoehe."""
    punktkennung: str
    """Punktkennung."""
    sonstigeEigenschaft: list[str]
    """Sonstige Eigenschaft."""
    umbauterRaum: float
    """Umbauter Raum."""
    weitereGebaeudefunktion: list[EnumPair]
    """Weitere Gebaeudefunktion."""
    zustand: EnumPair
    """Zustand."""


class PartProps(Object):
    """Descriptive properties of a Part, as defined in GeoInfoDok."""

    abbaugut: EnumPair
    """Abbaugut."""
    ackerzahlOderGruenlandzahl: str
    """Ackerzahl oder Gruenlandzahl."""
    anzahlDerFahrstreifen: int
    """Anzahl der Fahrstreifen."""
    anzahlDerStreckengleise: EnumPair
    """Anzahl der Streckengleise."""
    art: EnumPair
    """Art."""
    artDerBebauung: EnumPair
    """Art der Bebauung."""
    artDerFestlegung: EnumPair
    """Art der Festlegung."""
    bahnkategorie: list[EnumPair]
    """Bahnkategorie."""
    bedeutung: list[EnumPair]
    """Bedeutung."""
    befahrbarkeit: EnumPair
    """Befahrbarkeit."""
    befestigung: EnumPair
    """Befestigung."""
    besondereFahrstreifen: EnumPair
    """Besondere Fahrstreifen."""
    besondereFunktion: EnumPair
    """Besondere Funktion."""
    besondereVerkehrsbedeutung: EnumPair
    """Besondere Verkehrsbedeutung."""
    bezeichnung: list[str]
    """Bezeichnung."""
    bodenart: EnumPair
    """Bodenart."""
    bodenstufe: EnumPair
    """Bodenstufe."""
    bodenzahlOderGruenlandgrundzahl: str
    """Bodenzahl oder Gruenlandgrundzahl."""
    bodenzahlOderGruenlandgrundzahlGrabloch: str
    """Bodenzahl oder Gruenlandgrundzahl (Grabloch)."""
    breiteDerFahrbahn: int
    """Breite der Fahrbahn."""
    breiteDesGewaessers: int
    """Breite des Gewaessers."""
    breiteDesVerkehrsweges: int
    """Breite des Verkehrsweges."""
    datumAbgabe: str
    """Datum-Abgabe."""
    datumAnordnung: str
    """Datum-Anordnung."""
    datumBesitzeinweisung: str
    """Datum-Besitzeinweisung."""
    datumDerLetztenUeberpruefung: str
    """Datum der letzten Ueberpruefung."""
    datumRechtskraeftig: str
    """Datum-rechtskraeftig."""
    datumrechtskraeftig: str
    """Datum-rechtskraeftig."""
    elektrifizierung: EnumPair
    """Elektrifizierung."""
    entstehungsart: list[EnumPair]
    """Entstehungsart."""
    entstehungsartOderKlimastufeWasserverhaeltnisse: list[EnumPair]
    """Entstehungsart oder Klimastufe/Wasserverhaeltnisse."""
    fahrbahntrennung: EnumPair
    """Fahrbahntrennung."""
    fahrtrichtung: bool
    """Fahrtrichtung."""
    fliessrichtung: bool
    """Fliessrichtung."""
    foerdergut: EnumPair
    """Foerdergut."""
    funktion: EnumPair
    """Funktion."""
    gewaesserkennzahl: str
    """Gewaesserkennzahl."""
    gewaesserkennziffer: str
    """Gewaesserkennziffer."""
    hydrologischesMerkmal: EnumPair
    """Hydrologisches Merkmal."""
    identnummer: str
    """Identnummer."""
    internationaleBedeutung: EnumPair
    """Internationale Bedeutung."""
    jahreszahl: int
    """Jahreszahl."""
    klassifizierung: EnumPair
    """Klassifizierung."""
    klimastufe: EnumPair
    """Klimastufe."""
    kulturart: EnumPair
    """Kulturart."""
    lagergut: EnumPair
    """Lagergut."""
    markierung: list[EnumPair]
    """Markierung."""
    merkmal: EnumPair
    """Merkmal."""
    nummer: str
    """Nummer."""
    nummerDerBahnstrecke: list[str]
    """Nummer der Bahnstrecke."""
    nummerDerLinie: list[str]
    """Nummer der Linie."""
    nummerDerSchutzzone: str
    """Nummer der Schutzzone."""
    nummerDesSchutzgebietes: str
    """Nummer des Schutzgebietes."""
    nutzung: list[EnumPair]
    """Nutzung."""
    nutzungsart: EnumPair
    """Nutzungsart."""
    oberflaechenmaterial: EnumPair
    """Oberflaechenmaterial."""
    primaerenergie: EnumPair
    """Primaerenergie."""
    rechtszustand: EnumPair
    """Rechtszustand."""
    schifffahrtskategorie: EnumPair
    """Schifffahrtskategorie."""
    seekennzahl: str
    """Seekennzahl."""
    sonstigeAngaben: list[EnumPair]
    """Sonstige Angaben."""
    spurweite: list[EnumPair]
    """Spurweite."""
    strassenschluessel: str
    """Strassenschluessel."""
    tagesabschnittsnummer: str
    """Tagesabschnittsnummer."""
    tidemerkmal: EnumPair
    """Tidemerkmal."""
    vegetationsmerkmal: EnumPair
    """Vegetationsmerkmal."""
    veraenderungOhneRuecksprache: bool
    """Veraenderung ohne Ruecksprache."""
    verkehrsbedeutungInneroertlich: EnumPair
    """Verkehrsbedeutung inneroertlich."""
    verkehrsbedeutungUeberoertlich: EnumPair
    """Verkehrsbedeutung ueberoertlich."""
    verkehrsdienst: EnumPair
    """Verkehrsdienst."""
    wasserspiegelhoeheInStehendemGewaesser: int
    """Wasserspiegelhoehe in stehendem Gewaesser."""
    wasserverhaeltnisse: EnumPair
    """Wasserverhaeltnisse."""
    widmung: EnumPair
    """Widmung."""
    zone: EnumPair
    """Zone."""
    zustand: EnumPair
    """Zustand."""
    zustandsstufe: EnumPair
    """Zustandsstufe."""
    zustandsstufeOderBodenstufe: EnumPair
    """Zustandsstufe oder Bodenstufe."""


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
    'befahrbarkeit': 'Befahrbarkeit',
    'befestigung': 'Befestigung',
    'beschaffenheit': 'Beschaffenheit',
    'besondereFahrstreifen': 'Besondere Fahrstreifen',
    'besondereFunktion': 'Besondere Funktion',
    'besondereVerkehrsbedeutung': 'Besondere Verkehrsbedeutung',
    'bezeichnung': 'Bezeichnung',
    'bodenart': 'Bodenart',
    'bodenstufe': 'Bodenstufe',
    'bodenzahlOderGruenlandgrundzahl': 'Bodenzahl oder Grünlandgrundzahl',
    'bodenzahlOderGruenlandgrundzahlGrabloch': 'Bodenzahl oder Grünlandgrundzahl (Grabloch)',
    'breiteDerFahrbahn': 'Breite der Fahrbahn',
    'breiteDesGewaessers': 'Breite des Gewässers',
    'breiteDesVerkehrsweges': 'Breite des Verkehrsweges',
    'dachart': 'Dachart',
    'dachform': 'Dachform',
    'dachgeschossausbau': 'Dachgeschossausbau',
    'datumAbgabe': 'Datum-Abgabe',
    'datumAnordnung': 'Datum-Anordnung',
    'datumBesitzeinweisung': 'Datum-Besitzeinweisung',
    'datumDerLetztenUeberpruefung': 'Datum der letzten Überprüfung',
    'datumRechtskraeftig': 'Datum-rechtskräftig',
    'datumrechtskraeftig': 'Datum-rechtskräftig',
    'durchfahrtshoehe': 'Durchfahrtshöhe',
    'elektrifizierung': 'Elektrifizierung',
    'entstehungsart': 'Entstehungsart',
    'entstehungsartOderKlimastufeWasserverhaeltnisse': 'Entstehungsart oder Klimastufe/Wasserverhältnisse',
    'fahrbahntrennung': 'Fahrbahntrennung',
    'fahrtrichtung': 'Fahrtrichtung',
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
    'klimastufe': 'Klimastufe',
    'kulturart': 'Kulturart',
    'lageZurErdoberflaeche': 'Lage zur Erdoberfläche',
    'lagergut': 'Lagergut',
    'markierung': 'Markierung',
    'merkmal': 'Merkmal',
    'name': 'Name',
    'nummer': 'Nummer',
    'nummerDerBahnstrecke': 'Nummer der Bahnstrecke',
    'nummerDerLinie': 'Nummer der Linie',
    'nummerDerSchutzzone': 'Nummer der Schutzzone',
    'nummerDesSchutzgebietes': 'Nummer des Schutzgebietes',
    'nutzung': 'Nutzung',
    'nutzungsart': 'Nutzungsart',
    'oberflaechenmaterial': 'Oberflächenmaterial',
    'objekthoehe': 'Objekthöhe',
    'primaerenergie': 'Primärenergie',
    'punktkennung': 'Punktkennung',
    'rechtszustand': 'Rechtszustand',
    'schifffahrtskategorie': 'Schifffahrtskategorie',
    'seekennzahl': 'Seekennzahl',
    'sonstigeAngaben': 'Sonstige Angaben',
    'sonstigeEigenschaft': 'Sonstige Eigenschaft',
    'spurweite': 'Spurweite',
    'strassenschluessel': 'Straßenschlüssel',
    'tagesabschnittsnummer': 'Tagesabschnittsnummer',
    'tidemerkmal': 'Tidemerkmal',
    'umbauterRaum': 'Umbauter Raum',
    'vegetationsmerkmal': 'Vegetationsmerkmal',
    'veraenderungOhneRuecksprache': 'Veränderung ohne Rücksprache',
    'verkehrsbedeutungInneroertlich': 'Verkehrsbedeutung innerörtlich',
    'verkehrsbedeutungUeberoertlich': 'Verkehrsbedeutung überörtlich',
    'verkehrsdienst': 'Verkehrsdienst',
    'wasserspiegelhoeheInStehendemGewaesser': 'Wasserspiegelhöhe in stehendem Gewässer',
    'wasserverhaeltnisse': 'Wasserverhältnisse',
    'weitereGebaeudefunktion': 'Weitere Gebäudefunktion',
    'widmung': 'Widmung',
    'zone': 'Zone',
    'zustand': 'Zustand',
    'zustandsstufe': 'Zustandsstufe',
    'zustandsstufeOderBodenstufe': 'Zustandsstufe oder Bodenstufe',
}

##


class DisplayTheme(gws.Enum):
    """Groups of related data that can be loaded and displayed with a Flurstueck."""

    lage = 'lage'
    """Location designations (Lage)."""
    gebaeude = 'gebaeude'
    """Buildings (Gebaeude)."""
    nutzung = 'nutzung'
    """Land use (Nutzung)."""
    festlegung = 'festlegung'
    """Legal and other designations (Festlegung)."""
    bewertung = 'bewertung'
    """Soil valuation (Bewertung)."""
    buchung = 'buchung'
    """Land register data (Buchung)."""
    eigentuemer = 'eigentuemer'
    """Owners (Eigentuemer)."""


EigentuemerAccessRequired = ['personName', 'personVorname']

BuchungAccessRequired = ['buchungsblattkennzeichenList']


class FlurstueckQueryOptions(gws.Data):
    """Options for a Flurstueck search."""

    strasseSearchOptions: gws.TextSearchOptions
    """How street names are matched."""
    nameSearchOptions: gws.TextSearchOptions
    """How person names are matched."""
    buchungsblattSearchOptions: gws.TextSearchOptions
    """How sheet identifiers are matched."""

    limit: int
    """Maximum number of results. A search with more results fails."""
    pageSize: int
    """Number of results per page."""
    offset: int
    """Number of results to skip."""
    sort: Optional[list[gws.SortOptions]]
    """Sort order, by columns of the Flurstueck index table."""

    displayThemes: list[DisplayTheme]
    """Related data to load with each Flurstueck."""

    withEigentuemer: bool
    """Whether owner data is accessed."""
    withBuchung: bool
    """Whether land register data is accessed."""
    withHistorySearch: bool
    """Whether historic objects are searched too."""
    withHistoryDisplay: bool
    """Whether historic objects and records are kept in the results."""


class FlurstueckQuery(gws.Data):
    """Flurstueck search criteria.

    Criteria that are set are combined with AND.
    """

    flurnummer: str
    """Flur number."""
    flurstuecksfolge: str
    """Flurstueck sequence number."""
    zaehler: str
    """Numerator of the Flurstueck number."""
    nenner: str
    """Denominator of the Flurstueck number."""
    flurstueckskennzeichen: str
    """Flurstueck identifier or its beginning."""

    flaecheBis: float
    """Maximum official area."""
    flaecheVon: float
    """Minimum official area."""

    gemarkung: str
    """Gemarkung name."""
    gemarkungCode: str
    """Gemarkung code. A code of up to 4 characters is prefixed with the Land code."""
    gemeinde: str
    """Gemeinde name."""
    gemeindeCode: str
    """Gemeinde code."""
    kreis: str
    """Kreis name."""
    kreisCode: str
    """Kreis code."""
    land: str
    """Land name."""
    landCode: str
    """Land code."""
    regierungsbezirk: str
    """Regierungsbezirk name."""
    regierungsbezirkCode: str
    """Regierungsbezirk code."""

    strasse: str
    """Street name."""
    hausnummer: str
    """House number, or ``*`` for any non-empty house number. Requires ``strasse``."""

    buchungsblattkennzeichenList: list[str]
    """Sheet identifiers, any of which must match."""

    personName: str
    """Owner last name or company name."""
    personVorname: str
    """Owner first name. Requires ``personName``."""

    shape: gws.Shape
    """Shape the Flurstueck must intersect."""

    uids: list[str]
    """Flurstueck uids."""

    options: Optional['FlurstueckQueryOptions']
    """Search options."""


class AdresseQueryOptions(gws.Data):
    """Options for an address search."""

    strasseSearchOptions: gws.TextSearchOptions
    """How street names are matched."""

    limit: int
    """Maximum number of results. A search with more results fails."""
    pageSize: int
    """Number of results per page."""
    offset: int
    """Number of results to skip."""
    sort: Optional[list[gws.SortOptions]]
    """Sort order, by columns of the address index table."""

    withHistorySearch: bool
    """Whether historic addresses are searched too."""


class AdresseQuery(gws.Data):
    """Address search criteria.

    Criteria that are set are combined with AND.
    """

    gemarkung: str
    """Gemarkung name."""
    gemarkungCode: str
    """Gemarkung code. A code of up to 4 characters is prefixed with the Land code."""
    gemeinde: str
    """Gemeinde name."""
    gemeindeCode: str
    """Gemeinde code."""
    kreis: str
    """Kreis name."""
    kreisCode: str
    """Kreis code."""
    land: str
    """Land name."""
    landCode: str
    """Land code."""
    regierungsbezirk: str
    """Regierungsbezirk name."""
    regierungsbezirkCode: str
    """Regierungsbezirk code."""

    strasse: str
    """Street name."""
    hausnummer: str
    """House number, or ``*`` for any non-empty house number. With ``bisHausnummer``, the lower bound of a range. Requires ``strasse``."""
    bisHausnummer: str
    """Upper bound of a house number range, inclusive. House numbers are compared by number, then by suffix; a bound without a suffix includes all suffixes, e.g. ``5`` includes ``5z``. Requires ``strasse``."""
    hausnummerNotNull: bool
    """Whether only addresses with a house number are found. Requires ``strasse``."""

    options: Optional['AdresseQueryOptions']
    """Search options."""


class IndexStatus(gws.Data):
    """Status of the ALKIS index, based on which index tables have data."""

    complete: bool
    """Whether all index tables have data."""
    basic: bool
    """Whether the basic index tables have data."""
    eigentuemer: bool
    """Whether the owner index tables have data."""
    buchung: bool
    """Whether the land register index tables have data."""
    missing: bool
    """Whether all index tables are empty or missing."""


##


class Reader:
    """Interface for reading ALKIS source data."""

    def read_all(self, cls: type, table_name: Optional[str] = None, uids: Optional[list[str]] = None):
        """Read source objects of a GeoInfoDok class.

        Args:
            cls: GeoInfoDok class from the ``geo_info_dok`` schema module.
            table_name: Source table name. Defaults to a name derived from the class.
            uids: Object identifiers to read. If empty, all objects are read.

        Returns:
            An iterable of ``cls`` instances.
        """

        pass

    def count(self, cls: type, table_name: Optional[str] = None) -> int:
        """Count source objects of a GeoInfoDok class.

        Args:
            cls: GeoInfoDok class from the ``geo_info_dok`` schema module.
            table_name: Source table name. Defaults to a name derived from the class.

        Returns:
            The number of objects.
        """

        pass


##
