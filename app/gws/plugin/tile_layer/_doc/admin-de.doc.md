# tile :/admin-de/config/layer/type/tile

%reference_de 'gws.plugin.tile_layer.layer.Config'

Rasterlayer, zeigt Kachel aus einer XYZ-Kachelquelle (wie z.B. OSM). Sie müssen die URL mit Platzhaltern ``{{{x}}}``, ``{{{y}}}`` und ``{{{z}}}`` angeben

```javascript

{
    "type": "title",
    "title": "Open Street Map",
    "display": "tile",
    "provider": {"url": "https://a.tile.openstreetmap.org/{{{z}}}/{{{x}}}/{{{y}}}.png"}
}
```