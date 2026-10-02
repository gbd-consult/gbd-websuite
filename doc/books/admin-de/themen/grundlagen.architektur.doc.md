# Architektur :/admin-de/themen/grundlagen/architektur

Die GBD WebSuite wird als Docker-Container ausgeliefert und läuft als Server, der auf HTTP(S) hört. Binden Sie QGIS-Projekte ein, kommt ein zweiter Container mit dem QGIS-Server hinzu. Der Container bringt alles Nötige mit; was Sie beisteuern, liegt in zwei eingebundenen Verzeichnissen – `/data` für Konfiguration und Daten, `/gws-var` für veränderliche Laufzeitdaten wie Caches und Sitzungen.

Auch wenn die WebSuite als reiner Webserver arbeiten kann, entfaltet sie ihren Nutzen erst im Zusammenspiel mit dem mitgelieferten Client, einer JavaScript-Anwendung, die im Browser eine interaktive Karte darstellt und dazu laufend Anfragen an den Server richtet.

## Prozesse

Innerhalb des Containers ist der Server kein einzelner Prozess, sondern ein Verbund: Ein Webserver nimmt die Anfragen entgegen, ein Applikationsserver führt die Logik der WebSuite aus, ein Kartenproxy bindet externe Quellen an und speichert Kacheln zwischen, ein Hintergrunddienst erledigt langlaufende Aufgaben wie den Druck, und eine Überwachung lädt die Konfiguration neu, sobald sich Dateien ändern.

Für die Konfiguration der Inhalte ist dieser Aufbau ohne Belang. Er wird dort wichtig, wo es um den [Betrieb](/admin-de/themen/betrieb/server) geht: Nicht benötigte Prozesse lassen sich abschalten, und Anzahl und Zeitlimits der Arbeitsprozesse bestimmen das Verhalten unter Last.

## Konfigurationsbaum

Das gesamte Verhalten der WebSuite ergibt sich aus einer einzigen Konfiguration, die als Baum aus verschachtelten Objekten aufgebaut ist. Das oberste Objekt ist die *Applikation*; darunter hängen die *Projekte* und alle weiteren Bausteine wie Layer, Aktionen und Datenbankverbindungen. Wie dieser Baum geschrieben wird, zeigt das [Konfigurationsmodell](/admin-de/themen/grundlagen/konfiguration).

## Webseiten und Projekte

Auf oberster Ebene trennt die WebSuite zwei Begriffe. Eine *Webseite* ist ein Hostname mit den zugehörigen Regeln für Auslieferung und Zugriff. Ein *Projekt* ist eine Karte mit ihren Einstellungen. Beide sind voneinander unabhängig: Dasselbe Projekt kann unter mehreren Webseiten erreichbar sein, und eine Webseite kann Inhalte ausliefern, die zu keinem Projekt gehören.

Daraus ergibt sich die Gliederung jeder Installation in zwei Ebenen – die globale Applikation und die einzelnen Projekte –, die die folgenden Abschnitte beschreiben.

## :/admin-de/themen/grundlagen/architektur/applikation
## :/admin-de/themen/grundlagen/architektur/projekte
