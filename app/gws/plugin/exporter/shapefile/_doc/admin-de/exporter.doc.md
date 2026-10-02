# Exporter "shapefile" :/admin-de/konfiguration/exporter/shapefile

Der Exporter `shapefile` wandelt Features in ein ESRI-Shapefile um. Es wird ein einzelner Vektor-Layer exportiert. Der Export nutzt den GDAL-Treiber `ESRI Shapefile`.

## Beispiel-Konfiguration ::

```javascript
exporters+ {
    type "shapefile"
    title "Shapefile"
    target "download"
}
```

Der Exporter wird über den `exporters`-Block der Konfiguration angemeldet. `title` erscheint im Client als Bezeichnung des Exportformats, `target "download"` liefert das Ergebnis als Datei-Download aus.

%ref "gws.plugin.exporter.shapefile.Config"
%demo "select_export"
