# Layer :/admin-de/themen/karten/layer

Ein Layer stellt eine Ebene der Karte dar. In seiner Konfiguration legen Sie einen Titel fest, der im Ebenenbaum erscheint, bestimmen das Erscheinungsbild und geben an, aus welcher Quelle das Kartenmaterial stammt. Ein Layer kann ein eigenes Ausmaß und eigene Zoomstufen definieren; liegt der Kartenmaßstab außerhalb dieses Bereichs, wird der Layer nicht angezeigt.

Grundsätzlich gibt es zwei Arten von Layern. Ein Rasterlayer liefert Geoinformation als fertiges Pixelbild und enthält selbst keine Sachdaten, lässt sich aber mit einer Suche verbinden. Ein Vektorlayer besteht aus einzelnen Objekten (Features) mit Geometrie und Attributen; sein Erscheinungsbild wird über Stilregeln festgelegt.

An anderer Stelle der Konfiguration – etwa in einer kombinierten Legende – verweisen Sie auf einen Layer über seine `uid`. Ohne Angabe erzeugt die WebSuite eine fortlaufende Nummer; vergeben Sie daher selbst eine eindeutige `uid`, wenn Sie den Layer ansprechen wollen.

## Rasterlayer

Für Rasterlayer bestimmen Sie unter anderem das Bildformat, in dem die Kacheln geliefert werden, den Anzeigemodus – ob der Server den ganzen Ausschnitt, einzelne Kacheln oder eine clientseitige Darstellung erzeugt – sowie das Zwischenspeichern der Bilder auf dem Server.

### Bildformat

Das Bildformat eines Rasterlayers legen Sie mit `imageFormat` fest. Ohne Angabe liefert der Layer PNG mit 8-Bit-Palette. In diesem Format werden die Bilder auch im Cache gespeichert. Von einem WMS oder QGIS Server fordert die WebSuite das passende Format an, sofern die Quelle es anbietet.

Sie wählen eines der vordefinierten Formate über `name`:

| `name` | Format |
|---|---|
| `PNG8` | PNG mit 8-Bit-Palette (Vorgabe) |
| `PNG24` | PNG in voller Farbtiefe mit Transparenz |
| `JPEG_LOW`, `JPEG`, `JPEG_HIGH` | JPEG mit Qualität 50, 75 und 90 |
| `WEBP` | WebP, verlustbehaftet |
| `WEBP_LOSSLESS` | WebP, verlustfrei |

Mit `quality` setzen Sie die Qualität selbst (JPEG 1-95, WebP 0-100), mit `background` die Farbe, mit der JPEG transparente Stellen füllt (Vorgabe Weiß). Ein Luftbild etwa wird als JPEG deutlich kleiner als als PNG:

```javascript
map.layers+ {
    title "Luftbild"
    type "wmsflat"
    provider.url "https://example.com/wms"
    sourceLayers.names [ "dop" ]
    imageFormat { name "JPEG" quality 85 }
}
```

Eigene Formatnamen gibt es nicht. Die Liste `imageFormats` eines [OWS-Dienstes](/admin-de/themen/publishing/ows) verwendet dieselben Einträge und bestimmt, welche Formate der Dienst anbietet.

## Vektorlayer

Für Vektorlayer bestimmen Sie das Datenmodell, das Aussehen der Features und ob und wie sie editiert werden dürfen. Über die Ladestrategie legen Sie fest, ob alle Features auf einmal oder nur die des sichtbaren Ausschnitts geladen werden.

Das Aussehen beschreiben Sie mit CSS: Füllung und Linien der Geometrie, Marker an den Stützpunkten sowie Schrift und Lage der Beschriftung. Die WebSuite versteht dafür einen Teil der Standard-CSS-Eigenschaften und ergänzt sie um eigene. Die Regeln geben Sie entweder unmittelbar in der Konfiguration an oder verweisen auf einen Selektor aus einer eigenen CSS-Datei; über den Maßstabsbereich einer Beschriftung steuern Sie zudem, ab wann sie erscheint. Die vollständige Liste der [CSS-Eigenschaften](/admin-de/konfiguration/style) gehört zur Konfigurationsreferenz.

%see
Siehe auch: [Konfiguration/Layer](/admin-de/konfiguration/layer), [Konfiguration/Modelle](/admin-de/konfiguration/model), [Konfiguration/CSS-Stile](/admin-de/konfiguration/style).
%end
