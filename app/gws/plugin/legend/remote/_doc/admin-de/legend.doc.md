# Legende "remote" :/admin-de/konfiguration/legend/remote

Die Legende `remote` bindet Legendenbilder von externen Adressen ein. Die Bild-URLs geben Sie über `urls` an; werden mehrere angegeben, werden sie zu einem Bild kombiniert.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "NRW ALKIS WMS"
    type "wmsflat"
    provider.url "https://www.wms.nrw.de/geobasis/wms_nw_alkis"
    sourceLayers.names [ "adv_alkis_gebaeude" ]
    legend {
        type "remote"
        urls [ "https://www.wms.nrw.de/geobasis/wms_nw_alkis?REQUEST=GetLegendGraphic&VERSION=1.3.0&FORMAT=image/png&LAYER=adv_alkis_gebaeude" ]
    }
}
```

Die Legende wird am Layer über den Block `legend` konfiguriert. `urls` enthält die Adressen der externen Legendenbilder, hier ein `GetLegendGraphic`-Aufruf des WMS-Dienstes. Bei mehreren URLs werden die Bilder zu einer Legende zusammengefügt.

%ref "gws.plugin.legend.remote.Config"
