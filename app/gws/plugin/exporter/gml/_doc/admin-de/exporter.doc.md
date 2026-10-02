# Exporter "gml" :/admin-de/konfiguration/exporter/gml

Der Exporter `gml` wandelt Features in eine GML-Datei um. Er unterstützt den Export mehrerer Vektor-Layer in einer Datei. Der Export nutzt den GDAL-Treiber `GML`.

## Beispiel-Konfiguration ::

```javascript
exporters+ {
    type "gml"
    title "GML (alle Layer in einer Datei)"
    target "download"
    withMultiLayer true
}
```

Der Exporter wird über den `exporters`-Block der Konfiguration angemeldet. `title` erscheint im Client als Bezeichnung des Exportformats, `target "download"` liefert das Ergebnis als Datei-Download aus. Mit `withMultiLayer true` werden mehrere Layer in eine einzige GML-Datei geschrieben statt in je eine Datei pro Layer.

%ref "gws.plugin.exporter.gml.Config"
%demo "select_export"
