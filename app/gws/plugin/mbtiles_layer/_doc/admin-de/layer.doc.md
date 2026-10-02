# Layer "mbtiles" :/admin-de/konfiguration/layer/mbtiles

Ein `mbtiles`-Layer stellt die Kacheln aus einer MBTiles-Datei als Rasterlayer dar. Die Datei geben Sie über den `provider` an. Mit `transparentColor` legen Sie eine als transparent zu behandelnde Farbe fest, mit `processing` steuern Sie zusätzliche Verarbeitungsdirektiven für die Rasterausgabe.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "TK-10"
    type "mbtiles"
    display "tile"
    provider.path "/data/mbtiles/TK10.mbtiles"
    transparentColor "#ffffff"
    processing [ "SCALE=0,240" ]
}
```

`provider.path` verweist auf die MBTiles-Datei. `display "tile"` gibt den Layer gekachelt und serverseitig zwischengespeichert aus. `transparentColor` behandelt reines Weiß als durchsichtig; die Direktive `SCALE=0,240` in `processing` zieht nahezu weiße Farbwerte auf Weiß, damit auch diese transparent werden.

%ref "gws.plugin.mbtiles_layer.layer.Config"
%demo "mbtiles_layer"
