# Layer "qgisflat" :/admin-de/konfiguration/layer/qgisflat

Ein `qgisflat`-Layer fasst einen oder mehrere Layer eines QGIS-Projekts zu einem einzelnen Rasterlayer zusammen. Das Projekt geben Sie über den `provider` an, mit `sourceLayers` wählen Sie die zu verwendenden Quelllayer aus. Mit `sqlFilters` hinterlegen Sie SQL-Filter je Quelllayer, die auf die Darstellung angewendet werden.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Bars und Kneipen"
    type "qgisflat"
    provider.path "/data/qgis/poi.qgs"
    sourceLayers.names [
        "Bar"
        "Kneipe"
    ]
}
```

`provider.path` benennt das QGIS-Projekt, `sourceLayers.names` wählt daraus die zu verwendenden Quelllayer über ihre Layernamen aus. Alle gewählten Layer werden zu einem einzelnen Rasterbild zusammengefasst. Ohne `sourceLayers` stellt der Layer das gesamte Projekt als ein Bild dar.

%ref "gws.plugin.qgis.flatlayer.Config"
%demo "qgis_dynamic_legend"
%demo "qgis_edit"
%demo "qgis_flat"
%demo "qgis_flat_partial"
