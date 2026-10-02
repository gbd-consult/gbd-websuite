# Dienste im Netz :/admin-de/themen/datenquellen/netzdienste

Fremde Daten müssen Sie nicht kopieren – die WebSuite fragt sie zur Laufzeit beim anbietenden Dienst ab, beschrieben im `provider` des Layers über dessen `url`. Was dabei für alle Dienste gleichermaßen gilt, von der Auswahl der Quell-Layer bis zur Achsen-Reihenfolge, behandelt das Thema [](/admin-de/themen/karten/quellen); hier geht es darum, welcher Layer-Typ zu welchem Dienst gehört.

Kartenbilder nach OGC-Standard liefert WMS, und zwar in zwei Formen: `wms` bildet den Baum des Dienstes als Gruppe von Einzellayern nach, `wmsflat` fasst mehrere Quell-Layer zu einem einzigen Rasterlayer zusammen, der über eine gemeinsame `GetMap`-Anfrage bezogen wird. Vorgerenderte Kacheln kommen über WMTS mit dem Typ `wmts`, bei dem Sie mit `sourceLayers` den Quell-Layer und mit `style` den WMTS-Stil wählen. Für Vektorobjekte gilt dieselbe Zweiteilung wie bei WMS: `wfs` als nachgebildeter Baum, `wfsflat` für genau einen Quell-Layer. WMS und WFS bringen zusätzlich je ein eigenes Modell und einen eigenen Finder mit.

Außerhalb der OGC-Standards sind Kacheldienste nach dem XYZ-Schema verbreitet, etwa OpenStreetMap. Der Layer-Typ `tile` bindet sie ein; seine `url` enthält die Platzhalter `{x}`, `{y}` und `{z}`. Mit `display` steuern Sie, ob der Server die Kacheln holt oder der Browser sie im Modus `client` unmittelbar abruft.

%see
Siehe auch: [Layer/wms](/admin-de/konfiguration/layer/wms), [Layer/wmsflat](/admin-de/konfiguration/layer/wmsflat), [Layer/wmts](/admin-de/konfiguration/layer/wmts), [Layer/wfs](/admin-de/konfiguration/layer/wfs), [Layer/wfsflat](/admin-de/konfiguration/layer/wfsflat), [Layer/tile](/admin-de/konfiguration/layer/tile).
%end
