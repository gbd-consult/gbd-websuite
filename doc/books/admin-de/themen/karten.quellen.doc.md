# Externe Kartendienste :/admin-de/themen/karten/quellen

Neben eigenen Daten kann die GBD WebSuite Karten und Objekte aus externen Diensten einbinden. Unterstützt werden die OGC-Dienste WMS und WMTS, Kacheldienste nach dem XYZ-Schema (etwa OpenStreetMap) sowie WFS für Vektordaten. Der jeweilige Layer-Typ bestimmt, welcher Dienst angesprochen wird.

Solche Dienste sind häufig nicht flach, sondern in mehreren Ebenen organisiert. Die WebSuite liest diese Struktur aus und stellt jede Ebene als *Quell-Layer* (SourceLayer) bereit, gekennzeichnet durch Name, Pfad innerhalb der Hierarchie und Tiefe. Aus einem Dienst müssen Sie daher nicht den gesamten Baum übernehmen: Über eine Auswahl der Quell-Layer – nach Namen, Pfadmuster oder Tiefe – binden Sie gezielt die benötigten Ebenen ein.

Alle Einstellungen für den Zugriff auf den Dienst fassen Sie im `provider`-Objekt des Layers zusammen.

## Geschützte Dienste

Verlangt ein Dienst eine Anmeldung, hinterlegen Sie die Zugangsdaten im `provider` unter `authorization`. Unterstützt wird HTTP-Basic (`type "basic"` mit `username` und `password`). Diese Anmeldung gilt für die Anfragen der WebSuite an den Dienst, nicht für die Nutzer des Clients.

## Zwischenspeicherung der Capabilities

Beim Einbinden liest die WebSuite das Capabilities-Dokument des Dienstes – die Beschreibung seiner Ebenen und Fähigkeiten – und speichert es zwischen (`capsCacheMaxAge`, Vorgabe ein Tag). Ändert sich die Struktur eines externen Dienstes, wirkt sich das daher unter Umständen erst nach Ablauf dieser Zeit aus; für einen sofortigen Effekt verkürzen Sie den Wert oder starten den Server neu.

## Koordinatensystem und Achsen-Reihenfolge

Rendert ein externer Dienst verdreht oder liefert leere Bilder, liegt das meist an der Achsen-Reihenfolge geografischer Koordinatensysteme (Lat/Lon gegenüber Lon/Lat, etwa bei `EPSG:4326`). Zwei Angaben im `provider` beheben das: `alwaysXY` erzwingt die Reihenfolge Lon/Lat, `forceCrs` bindet den Dienst fest an ein bestimmtes Koordinatensystem. Bei WMS regelt zusätzlich `bottomFirst`, ob die Ebenen des Dienstes von unten nach oben aufgeführt sind. Mit `maxRequests` begrenzen Sie die Zahl gleichzeitiger Anfragen an einen empfindlichen Dienst.

%see
Siehe auch: [Konfiguration/Layer](/admin-de/konfiguration/layer).
%end
