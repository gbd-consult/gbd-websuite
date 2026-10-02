# Exporter "csv" :/admin-de/konfiguration/exporter/csv

Der Exporter `csv` wandelt Features in eine CSV-Datei um. Die Ausgabe erfolgt als tabellarischer Text mit einer Zeile je Feature; es wird ein einzelner Vektor-Layer exportiert. Der Export nutzt den GDAL-Treiber `CSV`.

## Beispiel-Konfiguration ::

```javascript
exporters+ {
    type "csv"
    title "CSV"
    target "download"
}
```

Der Exporter wird über den `exporters`-Block der Konfiguration angemeldet. `title` erscheint im Client als Bezeichnung des Exportformats, `target "download"` liefert das Ergebnis als Datei-Download aus.

%ref "gws.plugin.exporter.csv.Config"
%demo "alkis_export"
