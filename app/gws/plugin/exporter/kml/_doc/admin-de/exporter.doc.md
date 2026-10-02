# Exporter "kml" :/admin-de/konfiguration/exporter/kml

Der Exporter `kml` wandelt Features in eine KML-Datei um, wie sie etwa von Google Earth gelesen wird. Er unterstützt den Export mehrerer Vektor-Layer in einer Datei. Der Export nutzt den GDAL-Treiber `KML`.

## Beispiel-Konfiguration ::

```javascript
exporters+ {
    type "kml"
    title "KML"
    target "download"
}
```

Der Exporter wird über den `exporters`-Block der Konfiguration angemeldet. `title` erscheint im Client als Bezeichnung des Exportformats, `target "download"` liefert das Ergebnis als Datei-Download aus. Sollen mehrere Layer in einer Datei zusammengefasst werden, setzen Sie `withMultiLayer true`.

%ref "gws.plugin.exporter.kml.Config"
%demo "select_export"
