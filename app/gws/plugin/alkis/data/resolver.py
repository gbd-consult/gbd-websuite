import os

import gws
import gws.types as t
import gws.lib.json2
from ..util import indexer
from ..util.connection import AlkisConnection

# table name, key column, value column

place_tables = {
    'land': 'ax_bundesland',
    'regierungsbezirk': 'ax_regierungsbezirk',
    'kreis': 'ax_kreisregion',
    'gemeinde': 'ax_gemeinde',
    'gemarkungsnummer': 'ax_gemarkung',
    'bezirk': 'ax_buchungsblattbezirk',
    'stelle': 'ax_dienststelle',
}

place_fields = {
    'land': ['land'],
    'regierungsbezirk': ['regierungsbezirk', 'land'],
    'kreis': ['kreis', 'regierungsbezirk', 'land'],
    'gemeinde': ['gemeinde', 'kreis', 'regierungsbezirk', 'land'],
    'gemarkungsnummer': ['gemarkungsnummer', 'land'],
    'bezirk': ['bezirk', 'land'],
    'stelle': ['stelle', 'land']
}

column_labels = {
    'abbaugut': u'Abbaugut',
    'abmarkung_marke': u'Abmarkung (Marke)',
    'administrativefunktion': u'administrative Funktion',
    'advstandardmodell': u'AdV Standard Modell',
    'anlass': u'Anlass',
    'anlassdesprozesses': u'Anlass des Prozesses',
    'anrede': u'Anrede',
    'anzahlderstreckengleise': u'Anzahl der Streckengleise',
    'archaeologischertyp': u'archäologischer Typ',
    'art': u'Art',
    'artderaussparung': u'Art der Aussparung',
    'artderbebauung': u'Art der Bebauung',
    'artderfestlegung': u'Art der Festlegung',
    'artderflurstuecksgrenze': u'Art der Flurstücksgrenze',
    'artdergebietsgrenze': u'Art der Gebietsgrenze',
    'artdergelaendekante': u'Art der Geländekante',
    'artdergeripplinie': u'Art der Geripplinie',
    'artdergewaesserachse': u'Art der Gewässerachse',
    'artdernichtgelaendepunkte': u'Art der Nicht Geländepunkte',
    'artderrechtsgemeinschaft': u'Art der Rechtsgemeinschaft',
    'artderstrukturierung': u'Art der Strukturierung',
    'artderverbandsgemeinde': u'Art der Verbandsgemeinde',
    'artdesmarkantenpunktes': u'Art des Markanten Punktes',
    'artdesnullpunktes': u'Art des Nullpunktes',
    'artdespolders': u'Art des Polders',
    'ausgabeform': u'Ausgabeform',
    'ausgabemedium': u'Ausgabemedium',
    'bahnhofskategorie': u'Bahnhofskategorie',
    'bahnkategorie': u'Bahnkategorie',
    'bauart': u'Bauart',
    'bauweise': u'Bauweise',
    'bauwerksfunktion': u'Bauwerksfunktion',
    'bedeutung': u'Bedeutung',
    'befestigung': u'Befestigung',
    'bemerkungzurabmarkung': u'Bemerkung zur Abmarkung',
    'berechnungsmethode': u'Berechnungsmethode',
    'berechnungsmethodehoehenlinie': u'Berechnungsmethode Höhenlinie',
    'beschaffenheit': u'Beschaffenheit',
    'besondereartdergewaesserbegrenzung': u'besondere Art der Gewässerbegrenzung',
    'besonderebedeutung': u'besondere Bedeutung',
    'besonderefahrstreifen': u'besondere Fahrstreifen',
    'besonderefunktion': u'besondere Funktion',
    'bewuchs': u'Bewuchs',
    'bezeichnungart': u'Bezeichnung (Art)',
    'blattart': u'Blattart',
    'bodenart': u'Bodenart',
    'buchungsart': u'Buchungsart',
    'dachform': u'Dachform',
    'dachgeschossausbau': u'Dachgeschossausbau',
    'darstellung': u'Darstellung',
    'dateityp': u'Datei Typ',
    'datenerhebung': u'Datenerhebung',
    'datenformat': u'Datenformat',
    'description': u'Description',
    'dimension': u'Dimension',
    'eigentuemerart': u'Eigentümerart',
    'elektrifizierung': u'Elektrifizierung',
    'entstehungsartoderklimastufewasserverhaeltnisse': u'Entstehungsart oder Klimastufe/Wasserverhältnisse',
    'fahrbahntrennung': u'Fahrbahntrennung',
    'foerdergut': u'Fördergut',
    'funktion': u'Funktion',
    'funktionhgr': u'Funktion HGR',
    'funktionoa': u'Funktion OA',
    'gebaeudefunktion': u'Gebäudefunktion',
    'genauigkeitsstufe': u'Genauigkeitsstufe',
    'geologischestabilitaet': u'geologische Stabilität',
    'gnsstauglichkeit': u'GNSS Tauglichkeit',
    'gruendederausgesetztenabmarkung': u'Gründe der ausgesetzten Abmarkung',
    'grundwasserschwankung': u'Grundwasserschwankung',
    'grundwasserstand': u'Grundwasserstand',
    'guetedesbaugrundes': u'Güte des Baugrundes',
    'guetedesvermarkungstraegers': u'Güte des Vermarkungsträgers',
    'hafenkategorie': u'Hafenkategorie',
    'hierarchiestufe3d': u'Hierarchiestufe3D',
    'hoehenstabilitaetauswiederholungsmessungen': u'Höhenstabilität aus Wiederholungsmessungen',
    'horizontaleausrichtung': u'horizontale Ausrichtung',
    'horizontfreiheit': u'Horizontfreiheit',
    'hydrologischesmerkmal': u'hydrologisches Merkmal',
    'identifikation': u'Identifikation',
    'impliziteloeschungderreservierung': u'implizite Löschung der Reservierung',
    'internationalebedeutung': u'internationale Bedeutung',
    'klassifizierung': u'Klassifizierung',
    'klassifizierunggr': u'Klassifizierung GR',
    'klassifizierungobg': u'Klassifizierung OBG',
    'konstruktionsmerkmalbauart': u'Konstruktionsmerkmal Bauart',
    'koordinatenstatus': u'Koordinatenstatus',
    'kulturart': u'Kulturart',
    'lagergut': u'Lagergut',
    'lagezurerdoberflaeche': u'Lage zur Erdoberfläche',
    'lagezuroberflaeche': u'Lage zur Oberfläche',
    'landschaftstyp': u'Landschaftstyp',
    'letzteabgabeart': u'Letzte Abgabe Art',
    'levelofdetail': u'level of detail',
    'liniendarstellung': u'Liniendarstellung',
    'markierung': u'Markierung',
    'merkmal': u'Merkmal',
    'messmethode': u'Messmethode',
    'nutzung': u'Nutzung',
    'oberflaechenmaterial': u'Oberflächenmaterial',
    'ordnung': u'Ordnung',
    'primaerenergie': u'Primärenergie',
    'produkt': u'Produkt',
    'punktart': u'Punktart',
    'punktstabilitaet': u'Punktstabilität',
    'punktvermarkung': u'Punktvermarkung',
    'qualitaetsangaben': u'Qualitätsangaben',
    'rechtszustand': u'Rechtszustand',
    'reservierungsart': u'Reservierungsart',
    'schifffahrtskategorie': u'Schifffahrtskategorie',
    'schwerestatus': u'Schwerestatus',
    'schweresystem': u'Schweresystem',
    'selektionsmassstab': u'Selektionsmassstab',
    'skizzenart': u'Skizzenart',
    'sonstigeangaben': u'Sonstige Angaben',
    'speicherinhalt': u'Speicherinhalt',
    'sportart': u'Sportart',
    'spurweite': u'Spurweite',
    'stellenart': u'Stellenart',
    'tidemerkmal': u'Tidemerkmal',
    'topographieundumwelt': u'Topographie und Umwelt',
    'ueberschriftimfortfuehrungsnachweis': u'Überschrift im Fortführungsnachweis',
    'ursprung': u'Ursprung',
    'vegetationsmerkmal': u'Vegetationsmerkmal',
    'verarbeitungsart': u'Verarbeitungsart',
    'verkehrsbedeutunginneroertlich': u'Verkehrsbedeutung Innerörtlich',
    'verkehrsbedeutungueberoertlich': u'Verkehrsbedeutung Ueberörtlich',
    'vermarkung_marke': u'Vermarkung (Marke)',
    'vermutetehoehenstabilitaet': u'Vermutete Höhenstabilität',
    'vertikaleausrichtung': u'Vertikale Ausrichtung',
    'vertrauenswuerdigkeit': u'Vertraünswürdigkeit',
    'verwendeteinstanzenthemen': u'Verwendete Instanzenthemen',
    'verwendeteobjekte': u'Verwendete Objekte',
    'verwendetethemen': u'Verwendete Themen',
    'weiteregebaeudefunktion': u'Weitere Gebäudefunktion',
    'wertigkeit': u'Wertigkeit',
    'widmung': u'Widmung',
    'wirtschaftsart': u'Wirtschaftsart',
    'zone': u'Zone',
    'zugriffsartfortfuehrungsanlass': u'Zugriffsart Fortführungsanlass',
    'zugriffsartproduktkennungbenutzung': u'Zugriffsart Produktkennung Benutzung',
    'zugriffsartproduktkennungfuehrung': u'Zugriffsart Produktkennung Führung',
    'zustand': u'Zustand',
    'zustandsstufeoderbodenstufe': u'Zustandsstufe oder Bodenstufe',

    'land': u'Bundesland',
    'regierungsbezirk': u'Regierungsbezirk',
    'kreis': u'Kreis',
    'gemeinde': u'Gemeinde',
    'bezirk': u'Buchungsblattbezirk',
    'stelle': u'Dienststelle'
}

# s. ALKIS_OK_6_0.html#_3DFA354A0193
# id - table - label
nutzung_tables = [
    (41001, 'ax_wohnbauflaeche', u'Wohnbaufläche'),
    (41002, 'ax_industrieundgewerbeflaeche', u'Industrie- und Gewerbefläche'),
    (41003, 'ax_halde', u'Halde'),
    (41004, 'ax_bergbaubetrieb', u'Bergbaubetrieb'),
    (41005, 'ax_tagebaugrubesteinbruch', u'Tagebau, Grube, Steinbruch'),
    (41006, 'ax_flaechegemischternutzung', u'Fläche gemischter Nutzung'),
    (41007, 'ax_flaechebesondererfunktionalerpraegung', u'Fläche besonderer funktionaler Prägung'),
    (41008, 'ax_sportfreizeitunderholungsflaeche', u'Sport-, Freizeit- und Erholungsfläche'),
    (41009, 'ax_friedhof', u'Friedhof'),
    (41010, 'ax_siedlungsflaeche', u'Siedlungsfläche'),
    (42001, 'ax_strassenverkehr', u'Straßenverkehr'),
    (42002, 'ax_strasse', u'Straße'),
    (42003, 'ax_strassenachse', u'Straßenachse'),
    (42005, 'ax_fahrbahnachse', u'Fahrbahnachse'),
    (42006, 'ax_weg', u'Weg'),
    (42008, 'ax_fahrwegachse', u'Fahrwegachse'),
    (42009, 'ax_platz', u'Platz'),
    (42010, 'ax_bahnverkehr', u'Bahnverkehr'),
    (42014, 'ax_bahnstrecke', u'Bahnstrecke'),
    (42015, 'ax_flugverkehr', u'Flugverkehr'),
    (42016, 'ax_schiffsverkehr', u'Schiffsverkehr'),
    (43001, 'ax_landwirtschaft', u'Landwirtschaft'),
    (43002, 'ax_wald', u'Wald'),
    (43003, 'ax_gehoelz', u'Gehölz'),
    (43004, 'ax_heide', u'Heide'),
    (43005, 'ax_moor', u'Moor'),
    (43006, 'ax_sumpf', u'Sumpf'),
    (43007, 'ax_unlandvegetationsloseflaeche', u'Unland/Vegetationslose Fläche'),
    (43008, 'ax_flaechezurzeitunbestimmbar', u'Fläche zur Zeit unbestimmbar'),
    (44001, 'ax_fliessgewaesser', u'Fließgewässer'),
    (44002, 'ax_wasserlauf', u'Wasserlauf'),
    (44003, 'ax_kanal', u'Kanal'),
    (44004, 'ax_gewaesserachse', u'Gewässerachse'),
    (44005, 'ax_hafenbecken', u'Hafenbecken'),
    (44006, 'ax_stehendesgewaesser', u'Stehendes Gewässer'),
    (44007, 'ax_meer', u'Meer'),
]

# 'category' key
# for other tables it's assumed to be 'funktion'

nutzung_keys = {
    41003: 'lagergut',
    41004: 'abbaugut',
    41005: 'abbaugut',
    43001: 'vegetationsmerkmal',
    43002: 'vegetationsmerkmal',
    43003: 'vegetationsmerkmal',
}

PROPS_INDEX = 'idx_resolver_props'
PLACE_INDEX = 'idx_resolver_place'


def _create_props_index(conn):
    props = gws.lib.json2.from_path(os.path.dirname(__file__) + '/nasprops.json')
    data = []

    for ident, value in props.items():
        # e.g.  ['ax_polder:artdespolders:1000', 'Sommerpolder']
        s = ident.split(':')
        data.append({
            'table_name': s[0],
            'column_name': s[1],
            'property_key': s[2],
            'property_value': value,
        })

    conn.create_index_table(PROPS_INDEX, '''
        id SERIAL PRIMARY KEY,
        table_name CHARACTER VARYING,
        column_name CHARACTER VARYING,
        property_key CHARACTER VARYING,
        property_value CHARACTER VARYING
    ''')
    conn.index_insert(PROPS_INDEX, data)
    conn.mark_index_table(PROPS_INDEX)


def _create_place_index(conn: AlkisConnection):
    data = []
    tables = conn.table_names(conn.data_schema)

    for key, table in place_tables.items():
        if table not in tables:
            continue

        fields = place_fields[key]

        for r in conn.select_from_ax(table, fields + ['bezeichnung']):
            data.append({
                'table_name': table,
                'place_key': ','.join('%s=%s' % (f, r[f]) for f in fields),
                'place_id': r[fields[0]],
                'place_name': r['bezeichnung']

            })

    conn.create_index_table(PLACE_INDEX, '''
        id SERIAL PRIMARY KEY,
        table_name CHARACTER VARYING,
        place_key CHARACTER VARYING,
        place_id CHARACTER VARYING,
        place_name CHARACTER VARYING
    ''')
    conn.index_insert(PLACE_INDEX, data)
    conn.mark_index_table(PLACE_INDEX)


def create_index(conn: AlkisConnection):
    if not indexer.check_version(conn, PROPS_INDEX):
        _create_props_index(conn)
    if not indexer.check_version(conn, PLACE_INDEX):
        _create_place_index(conn)


def index_ok(conn: AlkisConnection):
    return indexer.check_version(conn, PROPS_INDEX) and indexer.check_version(conn, PLACE_INDEX)


def _load_props_for_table(conn: AlkisConnection, table):
    idx = conn.index_schema
    d = {}
    sql = f'SELECT * FROM {idx}.{PROPS_INDEX} WHERE table_name=%s'
    for r in conn.select(sql, [table]):
        c = r['column_name']
        d.setdefault(c, {})[r['property_key']] = r['property_value']
    return d


def _load_places(conn: AlkisConnection):
    idx = conn.index_schema
    d = {}
    rs = conn.select(f'SELECT place_key, place_name FROM {idx}.{PLACE_INDEX}')
    for r in rs:
        d[r['place_key']] = r['place_name']
    return d


def attributes(conn: AlkisConnection, table, rec):
    cc = gws.get_global(
        'alkis_resolver_props_' + table,
        lambda: _load_props_for_table(conn, table))

    attr = {}

    for name in cc:
        if name not in rec:
            continue
        v = cc[name].get(str(rec[name]))
        if v is not None:
            attr[name] = v
            attr[name + '_id'] = rec[name]

    return attr


def places(conn: AlkisConnection, rec):
    cc = gws.get_global(
        'alkis_resolver_places',
        lambda: _load_places(conn))

    attr = {}

    for name, fields in place_fields.items():
        key = ','.join('%s=%s' % (f, rec.get(f, '')) for f in fields)
        if key in cc:
            s = 'gemarkung' if name == 'gemarkungsnummer' else name
            attr[s] = cc[key]
            attr[s + '_id'] = rec[name]

    return attr


def nutzung_key(type_id, rec):
    key = nutzung_keys.get(type_id, 'funktion')

    if rec.get(key):
        return {
            'key': rec[key],
            'key_id': rec[key + '_id'],
            'key_label': column_labels[key]
        }
