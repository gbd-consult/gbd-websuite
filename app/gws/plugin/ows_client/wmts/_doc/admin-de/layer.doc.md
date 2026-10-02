# Layer "wmts" :/admin-de/konfiguration/layer/wmts

Ein `wmts`-Layer bindet einen Quelllayer eines WMTS-Dienstes als gekacheltes Rasterlayer ein. Den Dienst geben Sie über den `provider` an, mit `sourceLayers` wählen Sie den Quelllayer aus. Über `style` legen Sie den zu verwendenden WMTS-Stil fest, mit `display` steuern Sie den Anzeigemodus.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Stadtplanwerk Ruhrgebiet"
    type "wmts"
    provider.url "https://geodaten.metropoleruhr.de/spw2"
    sourceLayers.names [ "spw2_light_plus" ]
    withCache true
}
```

`provider.url` verweist auf den WMTS-Dienst, `sourceLayers.names` wählt daraus den anzuzeigenden Quelllayer aus. Ohne diese Angabe verwendet der Layer den ersten Quelllayer des Dienstes. `withCache true` speichert die Kacheln auf dem Server zwischen, was besonders bei einer Umprojektion in ein anderes Koordinatensystem die Antwortzeiten verbessert.

%ref "gws.plugin.ows_client.wmts.layer.Config"
%demo "wmts_layer"
%demo "wmts_reprojected"
%demo "wmts_source"
