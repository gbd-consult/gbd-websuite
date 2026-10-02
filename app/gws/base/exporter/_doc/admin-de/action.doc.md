# Aktion "exporter" :/admin-de/konfiguration/action/exporter

Die Aktion `exporter` stellt die Schnittstelle für den Export von Objektdaten bereit. Sie startet und überwacht Export-Jobs im Hintergrund und liefert die erzeugten Dateien zum Download aus.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "exporter"
}

exporters+ {
    type "shapefile"
    title "Shapefile"
    target "download"
    access "allow all"
}

exporters+ {
    type "gml"
    title "GML (alle Layer in einer Datei)"
    target "download"
    access "allow all"
    withMultiLayer true
}
```

Die Aktion selbst hat keine eigenen Optionen; sie schaltet nur die Export-Schnittstelle frei. Die konkreten Ausgabeformate konfigurieren Sie getrennt als `exporters`. Über `type` wählen Sie das Format (u. a. `shapefile`, `gml`, `kml`, `geojson`), `target "download"` liefert das Ergebnis als Datei aus. Mit `withMultiLayer true` werden mehrere Layer in eine gemeinsame Datei geschrieben statt je Layer eine eigene.

%ref "gws.base.exporter.action.Config"
%demo "select_export"
