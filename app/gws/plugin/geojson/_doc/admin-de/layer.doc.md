# Layer "geojson" :/admin-de/konfiguration/layer/geojson

Ein `geojson`-Layer stellt die Features aus einer GeoJSON-Quelle als Vektorlayer dar. Die Quelle geben Sie über den `provider` an; dessen Geometrien und Attribute stehen anschließend für Darstellung, Suche und Datenmodelle zur Verfügung.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Bank"
    type "geojson"
    provider.path "/demos/poi/poi.bank.geojson"
}

map.layers+ {
    title "Districts"
    type "geojson"
    provider.path "/demos/districts.geojson"
}
```

Beide Layer beziehen ihre Features aus einer GeoJSON-Datei, die `provider.path` benennt. Jeder `map.layers+`-Eintrag erzeugt einen eigenen Vektorlayer. Weicht das Koordinatensystem der Quelle vom `crs` der Karte ab, projiziert die WebSuite die Geometrien automatisch um.

%ref "gws.plugin.geojson.layer.Config"
%demo "geojson_layer"
%demo "geojson_reprojected"
