const CATS = [
  { "category_id": 1,  "category_name": "Naturflächen", "color": "#317234" },
  { "category_id": 2,  "category_name": "Siedlungsgebiete", "color": "#7A6E6B" },
  { "category_id": 3,  "category_name": "Gebäude", "color": "#5C4840" },
  { "category_id": 4,  "category_name": "Verkehrsflächen", "color": "#676767" },
  { "category_id": 5,  "category_name": "Landwirtschaft und Industrie", "color": "#7D8321" },
  { "category_id": 6,  "category_name": "Elektrizität", "color": "#A48C22" },
  { "category_id": 7,  "category_name": "Freizeit und Erholung", "color": "#196C64" },
  { "category_id": 8,  "category_name": "Sport", "color": "#A6492C" },
  { "category_id": 9,  "category_name": "Besondere Einrichtungen", "color": "#52397E" },
  { "category_id": 10, "category_name": "Militär", "color": "#37461F" },
  { "category_id": 11, "category_name": "Sonstiges", "color": "#7B7B7B" },
  { "category_id": 12, "category_name": "Fahrstraßen", "color": "#952522" },
  { "category_id": 13, "category_name": "Innerörtliche Straßen", "color": "#A35B00" },
  { "category_id": 14, "category_name": "Straßen und Wege für unmotorisierten Verkehr", "color": "#9C405F" },
  { "category_id": 15, "category_name": "Feld- und Waldwege", "color": "#695853" },
  { "category_id": 16, "category_name": "Sonstige Wege", "color": "#8C8582" },
  { "category_id": 17, "category_name": "Eisenbahnen", "color": "#2D3A41" },
  { "category_id": 18, "category_name": "Seilbahnen", "color": "#4E5E65" },
  { "category_id": 19, "category_name": "Wasserwege", "color": "#026595" },
  { "category_id": 20, "category_name": "Schiffsverkehr", "color": "#014D7B" },
  { "category_id": 21, "category_name": "Steige", "color": "#47312A" },
  { "category_id": 22, "category_name": "Infrastruktur Energieversorgung", "color": "#A37D1D" },
  { "category_id": 23, "category_name": "Barrieren", "color": "#3F3F3F" },
  { "category_id": 24, "category_name": "Naturmerkmale", "color": "#1E5120" },
  { "category_id": 25, "category_name": "Landes- und Verwaltungsgrenzen", "color": "#6F2E7A" },
  { "category_id": 26, "category_name": "Lokale", "color": "#992A4F" },
  { "category_id": 27, "category_name": "Kultur, Unterhaltung und Kunst", "color": "#651972" },
  { "category_id": 28, "category_name": "Historische Objekte", "color": "#4F372F" },
  { "category_id": 29, "category_name": "Freizeit, Erholung und Sport", "color": "#00707D" },
  { "category_id": 30, "category_name": "Reststoffverwertung", "color": "#544D0F" },
  { "category_id": 31, "category_name": "Outdoor", "color": "#446724" },
  { "category_id": 32, "category_name": "Tourismus und Beherbergung", "color": "#9F3514" },
  { "category_id": 33, "category_name": "Finanzwesen", "color": "#005950" },
  { "category_id": 34, "category_name": "Gesundheitswesen", "color": "#8C123E" },
  { "category_id": 35, "category_name": "Kommunikation", "color": "#3C467D" },
  { "category_id": 36, "category_name": "Verkehr", "color": "#4C4C4C" },
  { "category_id": 37, "category_name": "Besondere Straßenpunkte / Barrieren", "color": "#2B2B2B" },
  { "category_id": 38, "category_name": "Naturmerkmale", "color": "#123D15" },
  { "category_id": 39, "category_name": "Verwaltungseinrichtungen", "color": "#252F6F" },
  { "category_id": 40, "category_name": "Andachtsstätten", "color": "#5C176E" },
  { "category_id": 41, "category_name": "Geschäfte und Dienstleistungen", "color": "#A67400" },
  { "category_id": 42, "category_name": "Diverse künstliche Einrichtungen und Orientierungspunkte, Türme und Masten", "color": "#5E6B71" },
  { "category_id": 43, "category_name": "Stromversorgung", "color": "#A26D18" },
  { "category_id": 44, "category_name": "Siedlungsplätze, Gebaudeeingänge", "color": "#816448" }
];

module.exports = v => {

    let res = {
        '.geoservices_description .head2': {
            marginBottom: 0,
        },
        '.geoservices_description .head3': {
            marginTop: v.UNIT2,
            marginBottom: 0,
            fontSize: v.SMALL_FONT_SIZE,
        },
        '.geoservices_description table': {
            marginTop: v.UNIT2,
        },
    }

    for (let cat of CATS) {
        res[`.geoservices_${cat.category_id}`] = {
            display: 'block',
            width: 'fit-content',
            padding: [0, v.UNIT],
            marginBottom: v.UNIT,
            borderRadius: v.UNIT,
            fontSize: v.TINY_FONT_SIZE,
            backgroundColor: cat.color,
            color: 'white',
            lineHeight: 1.2,
        }
    }

    return res
}
