# Kommandozeilen-Befehl "qgis" :/admin-de/konfiguration/cli/qgis

Die Befehlsgruppe `qgis` unterstützt die Arbeit mit QGIS-Projekten.

`gws qgis caps --src <Projekt> --out caps.json` gibt den Aufbau eines Projekts als JSON aus.

`gws qgis copy --src <Quelle> --dst <Ziel>` kopiert ein Projekt zwischen Ablagen. Quelle und Ziel sind entweder ein Dateipfad oder eine Datenbank-Adresse in der Form `postgres:<dbUid>/<schema>/<projektname>` – so verschieben Sie ein Projekt zwischen Datei und Datenbank (siehe [Layer "qgis"](/admin-de/konfiguration/layer/qgis)).
