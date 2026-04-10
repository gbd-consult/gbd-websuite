# Karten :/admin-de/themen/karten

%reference_de 'gws.base.map.core.Config'

Eine Karte ist in der GBD WebSuite im Kern eine geordnete Sammlung von Layern (``layers``), die in einem definierten Koordinatenreferenzsystem dargestellt werden. Pro Projekt gibt es eine Hauptkarte und optional eine Übersichtskarte (``overviewMap``).

Die Kartenfunktionalität wird über die Aktion ``map`` freigeschaltet:

```javascript
{
    actions+ {
        type map
    }
    map {
        crs "EPSG:25832"
        ...
    }
}
```

### Haupt- und Übersichtskarte :/admin-de/themen/karten/uebersicht

Die **Hauptkarte** (``map``) ist die im Client angezeigte Arbeitskarte. Daneben kann optional eine **Übersichtskarte** (``overviewMap``) konfiguriert werden – eine zweite, eigenständige Karteninstanz, die typischerweise einen größeren räumlichen Überblick bietet und im Client als Miniaturkarte eingeblendet wird.

Beide verwenden dieselbe Konfigurationsstruktur:

```javascript
{
    map {
        crs "EPSG:25832"
        extent [280000, 5200000, 920000, 6100000]
        layers+ { ... }
    }
    overviewMap {
        crs "EPSG:25832"
        extent [50000, 5000000, 1150000, 6500000]
        layers+ { ... }
    }
}
```

### Koordinatenreferenzsystem :/admin-de/themen/karten/crs

Das Koordinatenreferenzsystem der Karte wird mit ``crs`` als EPSG-String angegeben. Der Standard ist ``EPSG:3857`` (Web Mercator), wie er von den meisten Kacheldiensten (OSM, Google Maps usw.) verwendet wird. Für Deutschland sind häufig verwendete Systeme ``EPSG:25832`` (UTM Zone 32N) oder ``EPSG:4326`` (WGS84).

Alle Layer im Projekt werden in diesem KBS dargestellt. Layer mit abweichenden Projektionen werden automatisch umprojiziert.

```javascript
{
    map {
        crs "EPSG:25832"
    }
}
```

%info
Derzeit kann pro Projekt nur ein einziges KBS definiert werden. Die Unterstützung mehrerer KBS pro Projekt ist für eine spätere Version geplant.
%end

### Kartenausmaß :/admin-de/themen/karten/extent

Mit ``extent`` definieren Sie die maximale räumliche Ausdehnung der Karte in KBS-Einheiten (für metrische Systeme in Metern, für geografische Systeme in Grad). Der Nutzer kann die Karte im Client nicht über diesen Bereich hinaus scrollen.

Die Koordinaten werden in der Reihenfolge ``[minX, minY, maxX, maxY]`` angegeben:

```javascript
{
    map {
        crs "EPSG:25832"
        extent [280000, 5200000, 920000, 6100000]
    }
}
```

Wird kein ``extent`` angegeben, berechnet die GBD WebSuite den Ausmaß automatisch aus den konfigurierten Layern.

Mit ``extentBuffer`` kann der automatisch berechnete Extent um einen festen Pixelwert vergrößert werden – nützlich, wenn Layer-Ausmaße zu knapp bemessen sind.

Die **Anfangsposition** beim Öffnen der Karte lässt sich mit ``center`` als Koordinatenpunkt im KBS der Karte festlegen. Fehlt ``center``, wird die Mitte des Extents verwendet.

```javascript
{
    map {
        crs "EPSG:25832"
        extent [280000, 5200000, 920000, 6100000]
        center [600000, 5700000]
    }
}
```

### Zoomstufen :/admin-de/themen/karten/zoom

%reference_de 'gws.gis.zoom.Config'

Über ``zoom`` kann die Anzahl und Aufteilung der verfügbaren Zoomstufen gesteuert werden. Zoomstufen lassen sich entweder als Maßstabszahlen (``scales``) oder als Auflösungswerte (``resolutions``) definieren.

**Maßstäbe** geben das Verhältnis von Bildschirmgröße zu Erdoberfläche an (z. B. ``10000`` bedeutet 1 mm am Bildschirm entspricht 10 m in der Realität). Da die physische Bildschirmgröße dem System nicht bekannt ist, gelten Maßstabsangaben als Näherungswerte; intern wird von ``1 Pixel = 0,28 × 0,28 mm`` (OGC-Standard) ausgegangen.

**Auflösungen** geben an, wie viele Meter pro Bildschirmpixel dargestellt werden.

```javascript
{
    map {
        zoom {
            scales [500000, 250000, 100000, 50000, 25000, 10000, 5000, 2500, 1000]
            initScale 100000
        }
    }
}
```

Alternativ mit Auflösungen:

```javascript
{
    map {
        zoom {
            resolutions [156543, 78271, 39135, 19567, 9783, 4891, 2445, 1222]
            initResolution 9783
        }
    }
}
```

Werden weder ``scales`` noch ``resolutions`` angegeben, verwendet die GBD WebSuite die Standard-Zoomstufen von OpenStreetMap (29 Stufen von ca. 156 m bis ca. 0,6 m pro Pixel).

Mit ``wrapX`` lässt sich die Karte horizontal endlos kacheln (Standardwert: ``false``).

### Layer :/admin-de/themen/karten/layer

Die Layer einer Karte werden über die ``layers``-Liste konfiguriert. Die Reihenfolge in der Konfiguration bestimmt sowohl die Darstellungsreihenfolge (erster Eintrag liegt oben) als auch die Reihenfolge im Ebenenbaum des Clients.

```javascript
{
    map {
        layers+ {
            type "wms"
            title "Hintergrundkarte"
            ...
        }
        layers+ {
            type "postgres"
            title "Fachdaten"
            ...
        }
    }
}
```

Layer können hierarchisch in Gruppen organisiert werden. Ausführliche Informationen zu den verfügbaren Layer-Typen und deren Konfiguration finden Sie im Thema [Layer](TODO LINK).
