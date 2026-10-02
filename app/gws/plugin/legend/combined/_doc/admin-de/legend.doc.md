# Legende "combined" :/admin-de/konfiguration/legend/combined

Die Legende `combined` fasst die Legenden mehrerer Layer zu einem Bild zusammen. Die einzubeziehenden Layer geben Sie über `layerUids` an; ihre Legenden werden nacheinander gerendert und kombiniert.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Versorgung"
    type "group"
    layers [
        { title "Straßen" type "qgis" uid "strassen" provider.path "/data/qgis/strassen.qgs" }
        { title "Gebäude" type "qgis" uid "gebaeude" provider.path "/data/qgis/gebaeude.qgs" }
    ]
    legend {
        type "combined"
        layerUids [ "strassen" "gebaeude" ]
    }
}
```

Die kombinierte Legende sitzt an der Gruppe und fasst die Legenden ihrer Teillayer zu einem Bild zusammen. `layerUids` listet die uids der Layer auf, deren Legenden zusammengeführt werden – hier die beiden Gruppenmitglieder `strassen` und `gebaeude`; sie werden in der angegebenen Reihenfolge gerendert.

%ref "gws.plugin.legend.combined.Config"
