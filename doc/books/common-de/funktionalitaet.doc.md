### Funktionalität :/common-de/funktionalitaet

Die GBD WebSuite stellt dem Anwender Kern- und Plugin-Funktionalität zur Verfügung, um Informationen in der Karte darzustellen, zu suchen, abzufragen, zu erstellen, zu bearbeiten und als druckfertige Karte auszugeben.

Aktuell umfasst die Funktionalität u.a.:

| Funktionalität				| Beschreibung							|
|-----------------------------------------------|---------------------------------------------------------------|
| QGIS und QGIS Server Integration		|QGIS Projekte können 1:1 dargestellt werden, QGIS Druckzusammenstellung inkl. Legende können genutzt werden und eine Vielzahl von in QGIS definierten Einstellungen (z.B. Legende oder Maßstabsfilter) können übernommen werden|
| Caching Getmap und GetPrint Anfragen 		|Druckdaten und Karten können bei Anfrage oder im Vorfeld gecached werden|
| Einbinden externer Dienste			|Externe Dienste können aus QGIS Projekten, über den GBD WebSuite Server oder direkt über den GBD WebSuite Client genutzt und bereitgestellt werden|
| Bereitstellung OGC-konformer Dienste		|Der GBD WebSuite Server kann OGC-konforme Dienste (WMS, WMTS, WFS, WCS) offen und mit Abfrage von Benutzerrechten bereitstellen|
| Bereitstellung INSPIRE-konformer Dienste	|Der GBD WebSuite Server kann Daten zu den Themen Bauleitplanung, Schulstandort und weiteres als INSPIRE-konforme Dienste (WMS, WMTS, WFS, WCS) offen und mit Abfrage von Benutzerrechten bereitstellen|
| Bereitstellung Catalogue Service for the Web	|Der GBD WebSuite Server unterstützt den Standard zum Offenlegen eines Katalogs von Geodatensätzen in XML im Internet (über HTTP(S)).
| Objektabfragen per Klick oder Mouseover	|Objekte in der Karte können per Klick oder Mouseover abgefragt werden|
| Suche	nach Inhalten (Attributen)		|Objekte können auf Basis ihrer Attribute über ein Suchfenster abgefragt, aufgelistet und im Kartenfenster dargestellt werden| 
| Räumliche Suche				|Informationen eines Layers können in der Karte mit Hilfe von Geometrien (Punkt, Linie, Fläche, Radius) abgefragt werden|
| Nominatim Suche				|Suche nach Orten auf Basis von OpenStreetMap (OSM) Daten über die Nominatim API
| Adressen Suchen über ALKIS			|Suche nach Adressen auf Basis von in PostGIS importierten Liegenschaftsdaten über eine QGIS Plugin|
| Markieren und Messen				|Positionen, Linien (Strecken) und Flächen können markiert, gemessen und individuell beschriftet werden|
| Bemaßung von Segmentlängen in der Karte	|Segmente in der Karte können automatisch mit den jeweiligen Streckenlängen und einer individuellen Beschriftung erstellt werden|
| Benutzer-Authentifizierung			|Abfrage von Benutzername und Passwort per LDAP, Postgres oder durch eine Anmeldedatei, um Rechte für das Arbeiten mit der GBD WebSuite zu erhalten|
| Daten und Objekte erstellen und editieren	|Über den WebGIS Client der GBD WebSuite könne Daten und Objekte editiert und neue Objekte hinzugefügt werden.|
| Beauskunftung Liegenschaften (ALKIS)		|Flurstücksuche auf Basis von amtlichen Liegschaftsdaten (ALKIS) inkl. Datenexport, Ausdruck sowie Datenschutzabfragen|
| Schnittstelle D-ProCon			|Werkzeug zur statistischen Analyse von Kartenobjekten mittels ALKIS Daten, um Informationen über die demografische Entwicklung zu erstellen|
| Schnittstelle GekoS Online			|Schnittstelle zum Einbinden der externen Fachschale GekoS Online in die GBD WebSuite|
| Unterstützung von Sprachen			|Die GBD WebSuite unterstützt aktuell die Sprachen Deutsch, Englisch, Georgisch und Tschechisch für den GBD WebSuite Client|
| Druckkkarten erstellen			|Druckkarten können auf QGIS Druckzusammenstellungen (.qpt) oder auf HTML-Vorlagen basieren und unterschiedliche Datenquellen miteinander kombinieren. Dabei wird auch das Drucken von auf dem Kartenfenster gezeichneten Objekten unterstützt (Redlining)
| Screenshot erstellen				|Kartenbereiche in variabler Auflösung als PNG-Screenshot speichern|



