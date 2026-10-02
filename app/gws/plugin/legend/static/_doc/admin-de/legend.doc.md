# Legende "static" :/admin-de/konfiguration/legend/static

Die Legende `static` bindet ein festes Bild als Legende ein. Die Bildquelle geben Sie über `path` an.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "OpenStreetMap"
    type "tile"
    provider.url "https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png"
    legend {
        type "static"
        path "/data/legends/osm_legend.png"
    }
}
```

Die Legende wird am Layer über den Block `legend` konfiguriert. `path` verweist auf die feste Bilddatei, die im Client als Legende dieses Layers angezeigt wird.

%ref "gws.plugin.legend.static.Config"
%demo "legend_static"
