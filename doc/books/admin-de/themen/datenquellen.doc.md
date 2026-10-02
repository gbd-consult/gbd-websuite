# Datenquellen :/admin-de/themen/datenquellen

Die GBD WebSuite bringt keine eigenen Geodaten mit, sondern greift auf vorhandene Bestände zu – auf eine Datei, eine Datenbank oder einen Dienst im Netz. Die Daten bleiben dabei an ihrem Platz und werden weiter dort gepflegt, wo sie herkommen; die WebSuite liest sie zur Laufzeit. Was sie unmittelbar lesen kann, muss weder konvertiert noch gespiegelt werden.

Eine Datenquelle ist kein eigenes Konfigurationsobjekt. Sie wird immer dort angegeben, wo sie gebraucht wird, und tritt dabei in bis zu drei Rollen auf: Ein [](/admin-de/themen/karten/layer) stellt sie in der Karte dar, ein [Modell](/admin-de/themen/daten/modelle) beschreibt ihre Attribute, ein Finder erschließt sie für die [](/admin-de/themen/darstellung/suche). In jeder Rolle trägt sie denselben Namen – zu einer PostGIS-Tabelle gehören zum Beispiel der Layer `postgres`, das Modell `postgres` und der Finder `postgres`.

Nicht jede Quelle deckt alle Rollen ab. Reine Rasterquellen wie MBTiles oder ein Kacheldienst liefern nur Bilder und kennen deshalb weder Modell noch Finder; Geokodierdienste wie Nominatim liefern umgekehrt nur Features und keine Kartenbilder. Ohne Datenquelle kommen zwei Typen aus: Der Layer-Typ `group` bündelt andere Layer zu einem Baumknoten, und der Modell-Typ `default` verwaltet Features aus den in der Konfiguration angegebenen Feldern.

Das Kapitel geht die unterstützten Quellen der Reihe nach durch: Dateien auf dem Server, Datenbanken, QGIS-Projekte, Dienste im Netz, Such- und Geokodierdienste sowie vorbereitete Fachdatenbestände.

## :/admin-de/themen/datenquellen/dateien
## :/admin-de/themen/datenquellen/datenbanken
## :/admin-de/themen/datenquellen/qgis
## :/admin-de/themen/datenquellen/netzdienste
## :/admin-de/themen/datenquellen/suchdienste
## :/admin-de/themen/datenquellen/fachdaten
