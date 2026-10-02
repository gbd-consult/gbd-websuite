# Applikation :/admin-de/konfiguration/application

Die Applikation ist das oberste Konfigurationsobjekt der GBD WebSuite. Sie bündelt alle globalen, serverweiten Einstellungen und enthält die Projekte. Einige Einstellungen – etwa `actions`, `finders`, `models` und `templates` – gelten global und lassen sich je Projekt erweitern.

## Projekte ::

Über `projects` geben Sie die Projekte als Liste an. Alternativ oder ergänzend laden Sie sie automatisch aus Verzeichnissen (`projectDirs`) oder über einzelne Dateipfade (`projectPaths`). Die Wege lassen sich kombinieren.

## Server und Web ::

`server` steuert die Serverprozesse und deren Einstellungen – Module, Anzahl der Worker und das Logging. `web` konfiguriert die Webseiten, die Auslieferung statischer und dynamischer Inhalte sowie das URL-Rewriting; dazu gehört auch die SSL-Einrichtung.

## Zugriff ::

`auth` bündelt die Authentifizierung – Anmeldemethoden, Provider und die Sitzungsverwaltung – sowie die rollenbasierte Zugriffskontrolle, mit der gesteuert wird, welche Nutzer auf welche Projekte, Layer und Funktionen zugreifen dürfen.

## Aktionen ::

`actions` schaltet die Server-Aktionen frei, die die Schnittstelle zwischen Client und Server bilden. Über sie laufen grundlegende Funktionen wie die Projektauslieferung, die Kartendarstellung, die Suche, das Editieren und der Druck. Global definierte Aktionen lassen sich je Projekt ergänzen.

## Daten ::

`database` definiert die Datenbankverbindungen (PostgreSQL/PostGIS), die von Layern, Suchen, Modellen und der Authentifizierung genutzt werden. `models` enthält global geltende Datenmodelle, `finders` die global geltenden Suchanbieter.

## Darstellung ::

`client` legt Aussehen und Verhalten der Client-Oberfläche global fest und ist je Projekt überschreibbar. `templates` enthält global geltende Vorlagen, `fonts` die Schriftarten für Kartenbeschriftungen, Feature-Labels und Druckausgaben.

## Ausgabe und Dienste ::

`printers` stellt Druckvorlagen für die PDF-Ausgabe bereit. `owsServices` konfiguriert die bereitgestellten OGC-Dienste wie WMS, WFS, WMTS und CSW.

## Weitere Einstellungen ::

`cache` steuert die Zwischenspeicherung von Kacheln und Dienstantworten, `storage` die serverseitige Datenablage. `locales` legt die verfügbaren Sprachen und Gebietsschemata fest, `metadata` die globalen Metadaten. `helpers` konfiguriert Hilfsobjekte, `developer` die Optionen für Entwicklung und Fehlersuche und `vars` globale Variablen zur Wiederverwendung in der Konfiguration.

## Beispiel-Konfiguration ::

```javascript
permissions.read "allow all"

actions+ { type "web" }
actions+ { type "project" }
actions+ { type "map" }
actions+ { type "search" }

auth.methods+ { type "web" }
auth.providers+ {
    type "file"
    path "/data/users.json"
}

database.providers+ {
    type "postgres"
    serviceName "gws"
}

client.elements+ { tag "Sidebar.Layers" }
client.elements+ { tag "Infobar.Scale" }

projects [
    @include /data/config/projects/stadtplan.cx
]
```

Diese Applikation ist bewusst knapp gehalten und zeigt das Zusammenspiel der globalen Bereiche. `permissions.read` setzt die Grundberechtigung. Die `actions`-Liste schaltet die serverweit verfügbaren Funktionen frei – von hier erben die Projekte. `auth` verbindet die Anmeldemethode (`web`, Formular mit Cookie) mit einem Provider (Benutzer aus einer Datei). `database.providers` definiert die Datenbankverbindung einmal global; Layer und Modelle sprechen sie später über ihre `uid` an. `client.elements` legt die Grundausstattung der Oberfläche fest, die Projekte ergänzen können. Die Projekte selbst werden über `@include` aus eigenen Dateien geladen.

%ref "gws.base.application.core.Config"
