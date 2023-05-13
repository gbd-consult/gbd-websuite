# tile :/admin-de/config/layer/type/

%reference_de 'gws.ext.layer.tile.Config'

Rasterlayer, zeigt Kachel aus einer XYZ-Kachelquelle (wie z.B. OSM). Sie müssen die URL mit Platzhaltern `{x}`, `{y}` und `{z}` angeben

    {
        "type": "title",
        "title": "Open Street Map",
        "url": "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"
    }

