import os

import gws
import gws.tools.json2

from ..util import indexer
from ..util.connection import AlkisConnection

VERSION = '71'

props_index = f'alkis_resolver_props_{VERSION}'
place_index = f'alkis_resolver_place_{VERSION}'


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
    'abbaugut': 'Abbaugut',
    'abmarkung_marke': 'Abmarkung (Marke)',
    'administrativefunktion': 'administrative Funktion',
    'advstandardmodell': 'AdV Standard Modell',
    'anlass': 'Anlass',
    'anlassdesprozesses': 'Anlass des Prozesses',
    'anrede': 'Anrede',
    'anzahlderstreckengleise': 'Anzahl der Streckengleise',
    'archaeologischertyp': 'archäologischer Typ',
    'art': 'Art',
    'artderaussparung': 'Art der Aussparung',
    'artderbebauung': 'Art der Bebauung',
    'artderfestlegung': 'Art der Festlegung',
    'artderflurstuecksgrenze': 'Art der Flurstücksgrenze',
    'artdergebietsgrenze': 'Art der Gebietsgrenze',
    'artdergelaendekante': 'Art der Geländekante',
    'artdergeripplinie': 'Art der Geripplinie',
    'artdergewaesserachse': 'Art der Gewässerachse',
    'artdernichtgelaendepunkte': 'Art der Nicht Geländepunkte',
    'artderrechtsgemeinschaft': 'Art der Rechtsgemeinschaft',
    'artderstrukturierung': 'Art der Strukturierung',
    'artderverbandsgemeinde': 'Art der Verbandsgemeinde',
    'artdesmarkantenpunktes': 'Art des Markanten Punktes',
    'artdesnullpunktes': 'Art des Nullpunktes',
    'artdespolders': 'Art des Polders',
    'ausgabeform': 'Ausgabeform',
    'ausgabemedium': 'Ausgabemedium',
    'bahnhofskategorie': 'Bahnhofskategorie',
    'bahnkategorie': 'Bahnkategorie',
    'bauart': 'Bauart',
    'bauweise': 'Bauweise',
    'bauwerksfunktion': 'Bauwerksfunktion',
    'bedeutung': 'Bedeutung',
    'befestigung': 'Befestigung',
    'bemerkungzurabmarkung': 'Bemerkung zur Abmarkung',
    'berechnungsmethode': 'Berechnungsmethode',
    'berechnungsmethodehoehenlinie': 'Berechnungsmethode Höhenlinie',
    'beschaffenheit': 'Beschaffenheit',
    'besondereartdergewaesserbegrenzung': 'besondere Art der Gewässerbegrenzung',
    'besonderebedeutung': 'besondere Bedeutung',
    'besonderefahrstreifen': 'besondere Fahrstreifen',
    'besonderefunktion': 'besondere Funktion',
    'bewuchs': 'Bewuchs',
    'bezeichnungart': 'Bezeichnung (Art)',
    'blattart': 'Blattart',
    'bodenart': 'Bodenart',
    'buchungsart': 'Buchungsart',
    'dachform': 'Dachform',
    'dachgeschossausbau': 'Dachgeschossausbau',
    'darstellung': 'Darstellung',
    'dateityp': 'Datei Typ',
    'datenerhebung': 'Datenerhebung',
    'datenformat': 'Datenformat',
    'description': 'Description',
    'dimension': 'Dimension',
    'eigentuemerart': 'Eigentümerart',
    'elektrifizierung': 'Elektrifizierung',
    'entstehungsartoderklimastufewasserverhaeltnisse': 'Entstehungsart oder Klimastufe/Wasserverhältnisse',
    'fahrbahntrennung': 'Fahrbahntrennung',
    'foerdergut': 'Fördergut',
    'funktion': 'Funktion',
    'funktionhgr': 'Funktion HGR',
    'funktionoa': 'Funktion OA',
    'gebaeudefunktion': 'Gebäudefunktion',
    'genauigkeitsstufe': 'Genauigkeitsstufe',
    'geologischestabilitaet': 'geologische Stabilität',
    'gnsstauglichkeit': 'GNSS Tauglichkeit',
    'gruendederausgesetztenabmarkung': 'Gründe der ausgesetzten Abmarkung',
    'grundwasserschwankung': 'Grundwasserschwankung',
    'grundwasserstand': 'Grundwasserstand',
    'guetedesbaugrundes': 'Güte des Baugrundes',
    'guetedesvermarkungstraegers': 'Güte des Vermarkungsträgers',
    'hafenkategorie': 'Hafenkategorie',
    'hierarchiestufe3d': 'Hierarchiestufe3D',
    'hoehenstabilitaetauswiederholungsmessungen': 'Höhenstabilität aus Wiederholungsmessungen',
    'horizontaleausrichtung': 'horizontale Ausrichtung',
    'horizontfreiheit': 'Horizontfreiheit',
    'hydrologischesmerkmal': 'hydrologisches Merkmal',
    'identifikation': 'Identifikation',
    'impliziteloeschungderreservierung': 'implizite Löschung der Reservierung',
    'internationalebedeutung': 'internationale Bedeutung',
    'klassifizierung': 'Klassifizierung',
    'klassifizierunggr': 'Klassifizierung GR',
    'klassifizierungobg': 'Klassifizierung OBG',
    'konstruktionsmerkmalbauart': 'Konstruktionsmerkmal Bauart',
    'koordinatenstatus': 'Koordinatenstatus',
    'kulturart': 'Kulturart',
    'lagergut': 'Lagergut',
    'lagezurerdoberflaeche': 'Lage zur Erdoberfläche',
    'lagezuroberflaeche': 'Lage zur Oberfläche',
    'landschaftstyp': 'Landschaftstyp',
    'letzteabgabeart': 'Letzte Abgabe Art',
    'levelofdetail': 'level of detail',
    'liniendarstellung': 'Liniendarstellung',
    'markierung': 'Markierung',
    'merkmal': 'Merkmal',
    'messmethode': 'Messmethode',
    'nutzung': 'Nutzung',
    'oberflaechenmaterial': 'Oberflächenmaterial',
    'ordnung': 'Ordnung',
    'primaerenergie': 'Primärenergie',
    'produkt': 'Produkt',
    'punktart': 'Punktart',
    'punktstabilitaet': 'Punktstabilität',
    'punktvermarkung': 'Punktvermarkung',
    'qualitaetsangaben': 'Qualitätsangaben',
    'rechtszustand': 'Rechtszustand',
    'reservierungsart': 'Reservierungsart',
    'schifffahrtskategorie': 'Schifffahrtskategorie',
    'schwerestatus': 'Schwerestatus',
    'schweresystem': 'Schweresystem',
    'selektionsmassstab': 'Selektionsmassstab',
    'skizzenart': 'Skizzenart',
    'sonstigeangaben': 'Sonstige Angaben',
    'speicherinhalt': 'Speicherinhalt',
    'sportart': 'Sportart',
    'spurweite': 'Spurweite',
    'stellenart': 'Stellenart',
    'tidemerkmal': 'Tidemerkmal',
    'topographieundumwelt': 'Topographie und Umwelt',
    'ueberschriftimfortfuehrungsnachweis': 'Überschrift im Fortführungsnachweis',
    'ursprung': 'Ursprung',
    'vegetationsmerkmal': 'Vegetationsmerkmal',
    'verarbeitungsart': 'Verarbeitungsart',
    'verkehrsbedeutunginneroertlich': 'Verkehrsbedeutung Innerörtlich',
    'verkehrsbedeutungueberoertlich': 'Verkehrsbedeutung Ueberörtlich',
    'vermarkung_marke': 'Vermarkung (Marke)',
    'vermutetehoehenstabilitaet': 'Vermutete Höhenstabilität',
    'vertikaleausrichtung': 'Vertikale Ausrichtung',
    'vertrauenswuerdigkeit': 'Vertraünswürdigkeit',
    'verwendeteinstanzenthemen': 'Verwendete Instanzenthemen',
    'verwendeteobjekte': 'Verwendete Objekte',
    'verwendetethemen': 'Verwendete Themen',
    'weiteregebaeudefunktion': 'Weitere Gebäudefunktion',
    'wertigkeit': 'Wertigkeit',
    'widmung': 'Widmung',
    'wirtschaftsart': 'Wirtschaftsart',
    'zone': 'Zone',
    'zugriffsartfortfuehrungsanlass': 'Zugriffsart Fortführungsanlass',
    'zugriffsartproduktkennungbenutzung': 'Zugriffsart Produktkennung Benutzung',
    'zugriffsartproduktkennungfuehrung': 'Zugriffsart Produktkennung Führung',
    'zustand': 'Zustand',
    'zustandsstufeoderbodenstufe': 'Zustandsstufe oder Bodenstufe',

    'land': 'Bundesland',
    'regierungsbezirk': 'Regierungsbezirk',
    'kreis': 'Kreis',
    'gemeinde': 'Gemeinde',
    'bezirk': 'Buchungsblattbezirk',
    'stelle': 'Dienststelle'
}

# s. ALKIS_6_0.html#_3B2A04300000

festlegung_tables = [
    (71001, 'ax_klassifizierungnachstrassenrecht', 'Klassifizierung nach Straßenrecht'),
    (71002, 'ax_anderefestlegungnachstrassenrecht', 'Andere Festlegung nach Straßenrecht'),
    (71003, 'ax_klassifizierungnachwasserrecht', 'Klassifizierung nach Wasserrecht'),
    (71004, 'ax_anderefestlegungnachwasserrecht', 'Andere Festlegung nach Wasserrecht'),
    (71005, 'ax_schutzgebietnachwasserrecht', 'Schutzgebiet nach Wasserrecht'),
    (71006, 'ax_naturumweltoderbodenschutzrecht', 'Natur-, Umwelt- oder Bodenschutzrecht'),
    (71007, 'ax_schutzgebietnachnaturumweltoderbodenschutzrecht', 'Schutzgebiet nach Natur-, Umwelt- oder Bodenschutzrecht'),
    (71008, 'ax_bauraumoderbodenordnungsrecht', 'Bau-, Raum- oder Bodenordnungsrecht'),
    (71009, 'ax_denkmalschutzrecht', 'Denkmalschutzrecht'),
    (71010, 'ax_forstrecht', 'Forstrecht'),
    (71011, 'ax_sonstigesrecht', 'Sonstiges Recht'),
    (71012, 'ax_schutzzone', 'Schutzzone'),
    (72001, 'ax_bodenschaetzung', 'Bodenschätzung'),
    (72002, 'ax_musterlandesmusterundvergleichsstueck', 'Muster-, Landesmuster- und Vergleichsstück'),
    (72003, 'ax_grablochderbodenschaetzung', 'Grabloch der Bodenschätzung'),
    (72004, 'ax_bewertung', 'Bewertung'),
    (72005, 'ax_kennziffergrabloch', 'KennzifferGrabloch'),
    (72006, 'ax_tagesabschnitt', 'Tagesabschnitt'),
]

# s. ALKIS_OK_6_0.html#_3DFA354A0193

nutzung_tables = [
    (41001, 'ax_wohnbauflaeche', 'Wohnbaufläche'),
    (41002, 'ax_industrieundgewerbeflaeche', 'Industrie- und Gewerbefläche'),
    (41003, 'ax_halde', 'Halde'),
    (41004, 'ax_bergbaubetrieb', 'Bergbaubetrieb'),
    (41005, 'ax_tagebaugrubesteinbruch', 'Tagebau, Grube, Steinbruch'),
    (41006, 'ax_flaechegemischternutzung', 'Fläche gemischter Nutzung'),
    (41007, 'ax_flaechebesondererfunktionalerpraegung', 'Fläche besonderer funktionaler Prägung'),
    (41008, 'ax_sportfreizeitunderholungsflaeche', 'Sport-, Freizeit- und Erholungsfläche'),
    (41009, 'ax_friedhof', 'Friedhof'),
    (41010, 'ax_siedlungsflaeche', 'Siedlungsfläche'),
    (42001, 'ax_strassenverkehr', 'Straßenverkehr'),
    (42002, 'ax_strasse', 'Straße'),
    (42003, 'ax_strassenachse', 'Straßenachse'),
    (42005, 'ax_fahrbahnachse', 'Fahrbahnachse'),
    (42006, 'ax_weg', 'Weg'),
    (42008, 'ax_fahrwegachse', 'Fahrwegachse'),
    (42009, 'ax_platz', 'Platz'),
    (42010, 'ax_bahnverkehr', 'Bahnverkehr'),
    (42014, 'ax_bahnstrecke', 'Bahnstrecke'),
    (42015, 'ax_flugverkehr', 'Flugverkehr'),
    (42016, 'ax_schiffsverkehr', 'Schiffsverkehr'),
    (43001, 'ax_landwirtschaft', 'Landwirtschaft'),
    (43002, 'ax_wald', 'Wald'),
    (43003, 'ax_gehoelz', 'Gehölz'),
    (43004, 'ax_heide', 'Heide'),
    (43005, 'ax_moor', 'Moor'),
    (43006, 'ax_sumpf', 'Sumpf'),
    (43007, 'ax_unlandvegetationsloseflaeche', 'Unland/Vegetationslose Fläche'),
    (43008, 'ax_flaechezurzeitunbestimmbar', 'Fläche zur Zeit unbestimmbar'),
    (44001, 'ax_fliessgewaesser', 'Fließgewässer'),
    (44002, 'ax_wasserlauf', 'Wasserlauf'),
    (44003, 'ax_kanal', 'Kanal'),
    (44004, 'ax_gewaesserachse', 'Gewässerachse'),
    (44005, 'ax_hafenbecken', 'Hafenbecken'),
    (44006, 'ax_stehendesgewaesser', 'Stehendes Gewässer'),
    (44007, 'ax_meer', 'Meer'),
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

def _create_props_index(conn):
    props = gws.tools.json2.from_path(os.path.dirname(__file__) + '/nasprops.json')
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

    conn.create_index_table(props_index, '''
        id SERIAL PRIMARY KEY,
        table_name CHARACTER VARYING,
        column_name CHARACTER VARYING,
        property_key CHARACTER VARYING,
        property_value CHARACTER VARYING
    ''')
    conn.index_insert(props_index, data)
    conn.mark_index_table(props_index)


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

    conn.create_index_table(place_index, '''
        id SERIAL PRIMARY KEY,
        table_name CHARACTER VARYING,
        place_key CHARACTER VARYING,
        place_id CHARACTER VARYING,
        place_name CHARACTER VARYING
    ''')
    conn.index_insert(place_index, data)
    conn.mark_index_table(place_index)


def create_index(conn: AlkisConnection):
    if not indexer.check_version(conn, props_index):
        _create_props_index(conn)
    if not indexer.check_version(conn, place_index):
        _create_place_index(conn)


def index_ok(conn: AlkisConnection):
    return indexer.check_version(conn, props_index) and indexer.check_version(conn, place_index)


def _load_props_for_table(conn: AlkisConnection, table):
    idx = conn.index_schema
    d = {}
    sql = f'SELECT * FROM {idx}.{props_index} WHERE table_name=%s'
    for r in conn.select(sql, [table]):
        c = r['column_name']
        d.setdefault(c, {})[r['property_key']] = r['property_value']
    return d


def _load_places(conn: AlkisConnection):
    idx = conn.index_schema
    d = {}
    rs = conn.select(f'SELECT place_key, place_name FROM {idx}.{place_index}')
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


def attributes_list(conn: AlkisConnection, table, rec):
    cc = gws.get_global(
        'alkis_resolver_props_' + table,
        lambda: _load_props_for_table(conn, table))

    attr = []

    for name in cc:
        if name not in rec:
            continue
        v = cc[name].get(str(rec[name]))
        if v is not None:
            attr.append({
                'name': name,
                'value_id': rec[name],
                'value': v,
                'label': column_labels.get(name, name)
            })

    return sorted(attr, key=lambda r: r['label'])


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
