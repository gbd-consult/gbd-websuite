# Legende "qgis" :/admin-de/konfiguration/legend/qgis

Die Legende `qgis` erzeugt die Legende aus einem QGIS-Projekt über den QGIS-Server (`GetLegendGraphic`). Das Projekt geben Sie über `provider` an; mit `sourceLayers` grenzen Sie die zu berücksichtigenden Quell-Layer ein.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Interessante Orte"
    type "qgisflat"
    provider.path "/data/qgis/poi.qgs"
    legend {
        type "qgis"
        sourceLayers.names [ "Interessante Orte" ]
        options.layerfontsize 12
    }
}
```

Die Legende wird am Layer über den Block `legend` konfiguriert. Fehlt ein eigener `provider`, wird das Projekt des Layers verwendet. `sourceLayers.names` grenzt die Legende auf bestimmte Quell-Layer ein, `options` reicht QGIS-spezifische Parameter der `GetLegendGraphic`-Anfrage durch, etwa die Schriftgröße der Layer-Titel.

%ref "gws.plugin.qgis.legend.Config"
