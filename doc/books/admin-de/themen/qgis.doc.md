# QGIS :/admin-de/themen/qgis

QGIS-Projekte sind einer der wichtigsten Wege, Karteninhalte in die GBD WebSuite zu bringen. Sie gestalten Ihre Karte im gewohnten Desktop-Werkzeug – Layer, Symbolisierung, Beschriftungen, Druckvorlagen – und binden das Ergebnis unverändert in ein Projekt der WebSuite ein. Ein Umbau der Daten ist dafür nicht nötig.

Ein QGIS-Projekt lässt sich auf drei Arten nutzen:

- als **vollständige Karte**: Der Layer-Typ [`qgis`](/admin-de/konfiguration/layer/qgis) übernimmt den Layerbaum des Projekts mitsamt seiner Struktur, Darstellung und Legenden.
- als **einzelner Layer**: Der Layer-Typ [`qgisflat`](/admin-de/konfiguration/layer/qgisflat) rendert eine Auswahl von Layern des Projekts zu einem einzigen flachen Bild, das sich in der Karte wie jeder andere Rasterlayer verhält.
- als **Druckvorlage**: Ein im Projekt angelegtes Drucklayout dient als [Vorlage](/admin-de/konfiguration/template/qgis) für die PDF-Ausgabe.

Darüber hinaus können die [Suche](/admin-de/konfiguration/finder/qgis) und die [Datenmodelle](/admin-de/konfiguration/model/qgis) unmittelbar auf die Quellen eines QGIS-Projekts zugreifen, und die [Legenden](/admin-de/konfiguration/legend/qgis) des Projekts lassen sich einbinden.

Zwischen Projekten der WebSuite und QGIS-Projekten besteht dabei keine feste Zuordnung. Ein WebSuite-Projekt kann mehrere QGIS-Projekte einbinden – etwa je eines für den Hintergrund, die Fachdaten und einen thematischen Layer. Umgekehrt kann dasselbe QGIS-Projekt von mehreren WebSuite-Projekten genutzt werden, jeweils mit unterschiedlicher Auswahl an Layern und unterschiedlichen Berechtigungen. Sie pflegen die Kartengrundlage damit an einer Stelle und leiten daraus beliebig viele Anwendungen ab.

Die Verbindung zum QGIS-Server und die gemeinsam genutzten Verzeichnisse gehören zur [Host-Konfiguration](/admin-de/themen/betrieb/host).

%see
Siehe auch: [Layer/qgis](/admin-de/konfiguration/layer/qgis), [Layer/qgisflat](/admin-de/konfiguration/layer/qgisflat).
%end
