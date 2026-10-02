# Einführung :/admin-de/einfuehrung

Die GBD WebSuite ist eine webbasierte GIS-Plattform, mit der Sie Karten, Geodaten und Fachanwendungen im Browser bereitstellen. Sie besteht aus einem Server und einem mitgelieferten JavaScript-Client. Dieses Handbuch richtet sich an Administratoren, die eine Installation einrichten und konfigurieren.

## Was die GBD WebSuite kann

- **Karten und Layer** aus verschiedenen Quellen kombinieren: QGIS-Projekte, PostGIS, WMS/WMTS/WFS, Kacheldienste (z. B. OpenStreetMap) und GeoJSON.
- **Suche** über mehrere Quellen (Nominatim, PostgreSQL, WFS …) mit einheitlicher Ergebnisdarstellung.
- **Editieren** von Vektordaten mit Formularen, Datenmodellen und Validierung.
- **Drucken** als PDF über HTML- oder QGIS-Druckvorlagen.
- **OWS-Dienste** (WMS, WFS, WMTS, CSW) bereitstellen – die WebSuite ist zugleich Client und Server.
- **Zugriffssteuerung** rollenbasiert über LDAP, Datenbank oder Dateien.

Der Funktionsumfang lässt sich über Plugins erweitern; Programmieren ist für den Betrieb nicht nötig.

## Architektur in Kürze

Die GBD WebSuite läuft als Docker-Container und hört auf HTTP(S). Sie verarbeitet zwei Arten von Anfragen:

- **statische Inhalte** (HTML, Bilder, PDF) wie ein gewöhnlicher Webserver,
- **dynamische Anfragen** (Kartenbilder, Suchergebnisse, Feature-Daten) über einen zentralen Endpunkt.

Jede dynamische Anfrage wird von einer *Server-Aktion* bearbeitet, die – je nach Anfrage – HTML, JSON oder PNG zurückgibt. Der mitgelieferte Client nutzt diese Aktionen, um eine interaktive Web-Karte darzustellen.

## Konfiguration

Das gesamte Verhalten der WebSuite wird über eine **Konfiguration** gesteuert – eine Baumstruktur aus Objekten in YAML oder JSON. Auf oberster Ebene stehen die *Applikation* (globale, serverweite Einstellungen) und die *Projekte* (je eine Karte mit den zugehörigen Layern, Suchen und Werkzeugen).

## Aufbau dieses Handbuchs

- [](/admin-de/erste-schritte) – WebSuite starten, installieren und ein erstes Projekt anlegen.
- [](/admin-de/themen) – Konzepte, Zusammenhänge und Konfiguration der einzelnen Bausteine.
- [](/admin-de/konfiguration) – die Konfigurationsbausteine nach Kategorien geordnet.
- [](/admin-de/reference) – vollständige, automatisch generierte Referenz aller Objekte.
