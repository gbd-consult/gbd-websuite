# Layer "raster" :/admin-de/konfiguration/layer/raster

Ein `raster`-Layer stellt eine oder mehrere Rasterdateien (etwa GeoTIFF) als Rasterlayer dar. Die Quelle geben Sie über den `provider` an. Über `sldPath` und `sldName` gestalten Sie den Layer mit einer SLD-Datei, mit `transparentColor` legen Sie eine als transparent zu behandelnde Farbe fest und mit `processing` zusätzliche Verarbeitungsdirektiven.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Luftbild"
    type "raster"
    display "tile"
    provider.paths [ "/data/raster/dop.tif" ]
    processing [ "BANDS=3" "GAMMA=0.7" ]
}
```

`provider.paths` listet die Rasterdateien auf; alternativ erfassen Sie mit `provider.pathPattern` ein ganzes Verzeichnis über ein Dateimuster. `display "tile"` gibt den Layer gekachelt aus. Die Direktiven in `processing` steuern die Rasterausgabe – hier die Beschränkung auf drei Bänder und eine Gamma-Korrektur. Für eine eigene Symbolisierung geben Sie über `sldPath` und `sldName` eine SLD-Datei an.

%ref "gws.plugin.raster_layer.layer.Config"
%demo "raster_layer"
%demo "raster_layer_dir"
