# Karten :/admin-de/konfiguration/map

Die [](/admin-de/themen/karten/karte) fasst die Layer eines Projekts in einem gemeinsamen Koordinatensystem zusammen und legt Ausmaß, Anfangsposition und Zoomstufen fest. Jedes Projekt hat eine Hauptkarte und optional eine Übersichtskarte; alle Layer werden unter einer gemeinsamen Wurzelgruppe zusammengefasst.

Mit `crs` legen Sie das Koordinatensystem fest (Vorgabe `EPSG:3857`), mit `extent` und `center` den Ausschnitt und den Mittelpunkt. Über `zoom` steuern Sie die verfügbaren Maßstäbe und Auflösungen. Die Karteninhalte selbst definieren Sie unter `layers`.

## Beispiel-Konfiguration ::

```javascript
map {
    crs 25832
    center [344371, 5677471]
    zoom {
        scales [500000, 250000, 100000, 50000, 25000, 10000, 5000, 2000, 1000]
        initScale 50000
    }
    layers+ {
        title "Basiskarte"
        type "qgis"
        provider.path "/data/qgis/basiskarte.qgs"
    }
    layers+ {
        title "Open Street Map"
        type "tile"
        provider.url "https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png"
    }
}
```

`crs` setzt das Koordinatensystem auf UTM Zone 32N (`EPSG:25832`), gebräuchlich für Lagedaten in Deutschland – alle Layer und Koordinatenangaben dieser Karte beziehen sich darauf. `center` gibt den Anfangsmittelpunkt in eben dieser Projektion an. Unter `zoom` legt `scales` die auswählbaren Maßstäbe fest, `initScale` den Maßstab beim Öffnen. Die `layers`-Liste bildet den Kartenbaum von oben nach unten: Die zuerst genannte Basiskarte liegt über dem OSM-Hintergrund.

%ref "gws.base.map.core.Config"
%demo "map_world_3857"
%demo "map_worlds_4326"
%demo "overview_map"
