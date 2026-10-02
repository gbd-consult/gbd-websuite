# Layer "wfs" :/admin-de/konfiguration/layer/wfs

Ein `wfs`-Layer bindet einen WFS-Dienst als Vektor-Layerbaum ein und bildet dessen Struktur automatisch als Gruppe von Einzellayern nach. Den Dienst geben Sie über den `provider` an. Über `rootLayers`, `excludeLayers` und `flattenLayers` steuern Sie, welche Quelllayer als Wurzeln dienen, welche ausgeschlossen und wie tief die Hierarchie zusammengefasst wird.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Digitale Verwaltungsgrenzen NRW"
    type "wfs"
    provider.url "https://www.wfs.nrw.de/geobasis/wfs_nw_dvg"
}
```

`provider.url` verweist auf den WFS-Dienst; die WebSuite liest dessen „feature types" aus und bildet sie automatisch als Gruppe einzelner Vektorlayer nach. Mit `rootLayers`, `excludeLayers` und `flattenLayers` schränken Sie die übernommenen Quelllayer ein oder fassen die Hierarchie flacher zusammen.

%ref "gws.plugin.ows_client.wfs.layer.Config"
%demo "wfs_layer"
