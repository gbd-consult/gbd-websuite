# Modell "geojson" :/admin-de/konfiguration/model/geojson

Der Modell-Typ `geojson` liest Features aus einer GeoJSON-Datei. Die Quelle geben Sie über den `provider` und dessen `path` an. Das Modell dient dem lesenden Zugriff auf die in der Datei enthaltenen Objekte.

## Beispiel-Konfiguration ::

```javascript
models+ {
    type "geojson"
    provider.path "/data/districts.geojson"
}
```

Das Modell liest die Features aus der über `provider.path` angegebenen GeoJSON-Datei. Die Felder werden aus den Eigenschaften der Datei abgeleitet; als Kennung dient `id`, als Geometriefeld `geometry`. Der Zugriff ist ausschließlich lesend.

%ref "gws.plugin.geojson.model.Config"
