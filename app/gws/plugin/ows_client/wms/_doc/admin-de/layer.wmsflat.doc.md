# Layer "wmsflat" :/admin-de/konfiguration/layer/wmsflat

Ein `wmsflat`-Layer fasst einen oder mehrere Quelllayer eines WMS-Dienstes zu einem einzelnen Rasterlayer zusammen, der über eine gemeinsame `GetMap`-Anfrage bezogen wird. Den Dienst geben Sie über den `provider` an, mit `sourceLayers` wählen Sie die zu verwendenden Quelllayer aus. Abfragbare Quelllayer werden automatisch für die Objektinformation genutzt.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "NRW ALKIS Flurstücke"
    type "wmsflat"
    provider.url "https://www.wms.nrw.de/geobasis/wms_nw_alkis"
    sourceLayers.names [ "adv_alkis_flurstuecke" ]
    display "tile"
}
```

`provider.url` benennt den WMS-Dienst, `sourceLayers.names` wählt die zusammenzufassenden Quelllayer aus; sie werden über eine gemeinsame `GetMap`-Anfrage zu einem einzelnen Rasterlayer gebündelt. Ohne `sourceLayers` verwendet der Layer alle Quelllayer. `display "tile"` bezieht das Bild gekachelt statt in einer einzigen großen Anfrage.

%ref "gws.plugin.ows_client.wms.flatlayer.Config"
%demo "wms_flat"
%demo "wms_flat_partial"
%demo "wms_flat_tiled"
