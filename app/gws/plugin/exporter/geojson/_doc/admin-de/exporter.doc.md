# Exporter "geojson" :/admin-de/konfiguration/exporter/geojson

Der Exporter `geojson` wandelt Features in eine GeoJSON-Datei um. Es wird ein einzelner Vektor-Layer exportiert, wobei Geometrie und Attribute jedes Features erhalten bleiben. Der Export nutzt den GDAL-Treiber `GeoJSON`.

## Beispiel-Konfiguration ::

```javascript
exporters+ {
    type "geojson"
    title "GeoJSON"
    target "download"
}
```

Der Exporter wird über den `exporters`-Block der Konfiguration angemeldet. `title` erscheint im Client als Bezeichnung des Exportformats, `target "download"` liefert das Ergebnis als Datei-Download aus.

%ref "gws.plugin.exporter.geojson.Config"
%demo "alkis_export"
%demo "select_export"
