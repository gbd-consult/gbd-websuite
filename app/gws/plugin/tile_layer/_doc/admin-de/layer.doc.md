# Layer "tile" :/admin-de/konfiguration/layer/tile

Ein `tile`-Layer bindet einen gekachelten Rasterdienst (XYZ) als Rasterlayer ein. Den Dienst geben Sie über den `provider` an, dessen `url` die Platzhalter `{x}`, `{y}` und `{z}` enthält. Mit `display` steuern Sie den Anzeigemodus; im Modus `client` werden die Kacheln direkt vom Browser abgerufen.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Open Street Map"
    type "tile"
    provider.url "https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png"
    display "tile"
    withCache true
}
```

`provider.url` gibt die XYZ-Vorlage mit den Platzhaltern `{z}`, `{x}` und `{y}` an. `display "tile"` bezieht die Kacheln über den Server; mit `withCache true` werden sie dort zwischengespeichert. Setzen Sie stattdessen `display "client"`, ruft der Browser die Kacheln direkt vom Dienst ab.

%ref "gws.plugin.tile_layer.layer.Config"
%demo "tile_client"
