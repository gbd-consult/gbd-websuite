# Layer "wms" :/admin-de/konfiguration/layer/wms

Ein `wms`-Layer bindet einen WMS-Dienst als Layerbaum ein und bildet dessen Struktur automatisch als Gruppe von Einzellayern nach. Den Dienst geben Sie über den `provider` an. Über `rootLayers`, `excludeLayers` und `flattenLayers` steuern Sie, welche Quelllayer als Wurzeln dienen, welche ausgeschlossen und wie tief die Hierarchie zusammengefasst wird.

## Beispiel-Konfiguration mit vollständigem Layerbaum ::

```javascript
map.layers+ {
    title "NRW ALKIS"
    type "wms"
    provider.url "https://www.wms.nrw.de/geobasis/wms_nw_alkis"
}
```

`provider.url` verweist auf den WMS-Dienst; die WebSuite übernimmt dessen Layerbaum unverändert und bildet jeden Quelllayer als eigenen WebSuite-Layer ab.

## Beispiel-Konfiguration mit angepasstem Layerbaum ::

```javascript
map.layers+ {
    title "NRW ALKIS"
    type "wms"
    provider.url "https://www.wms.nrw.de/geobasis/wms_nw_alkis"

    rootLayers.names [
        "adv_alkis_gebaeude"
        "adv_alkis_flurstuecke"
    ]
    flattenLayers.level 2
}
```

`rootLayers.names` bestimmt, welche Quelllayer als Wurzeln des Baums dienen und in welcher Reihenfolge sie erscheinen. `flattenLayers.level` fasst tiefer liegende Gruppen ab der angegebenen Ebene zu einzelnen Layern zusammen. Ergänzend schließt `excludeLayers` einzelne Quelllayer aus.

%ref "gws.plugin.ows_client.wms.layer.Config"
%demo "wms_layer"
%demo "wms_layer_flatten"
%demo "wms_roots"
%demo "wms_testbed"
