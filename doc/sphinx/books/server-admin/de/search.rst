Suche
======

Die GBD WebSuite implementiert eine einheitliche Suchfunktionalität, die mit verschiedenen Arten von Raum- und Domänendaten arbeiten kann. Grundsätzlich enthält eine Suchanfrage an den Server diese drei Parameter:

TABLE
    *Stichwort* ~ Stichwort, nach dem gesucht werden soll
    *shape* ~ eine Geometrie, falls vorhanden, ist die Suche räumlich auf diese Geometrie beschränkt.
    *Layers* ~ eine Liste der Ebenen, auf denen die Suche durchgeführt werden soll
/TABLE

Wenn der GBD WebSuite Server eine Suchanfrage erhält, verbindet er sich mit den konfigurierten Datenquellen und verwendet automatisch die für jede Quelle (oder *Provider*) gültige Methode. Beispielsweise werden für WMS-Quellen "GetFeatureInfo"-Anfragen gestellt, für Datenbankdaten wird eine "SELECT"-Abfrage durchgeführt und so weiter. Sobald der Server Ergebnisse aus verschiedenen Quellen erhält, werden sie konsolidiert, neu formatiert (siehe :doc:`features`) und als Liste einheitlicher GeoJSON-Features an den Client zurückgegeben.

Anbieter suchen
----------------

Nominatim
~~~~~~~~~

Eine Schnittstelle zu `Nominatim <https://nominatim.openstreetmap.org//>`_, der OSM-Suchmaschine existiert ebenfalls. Sie können die Parameter ``Land`` und ``Sprache`` konfigurieren, um die Suchergebnisse anzupassen.

SQL
~~~

Die GBD WebSuite ist in der Lage direkte Suchen in SQL (PostGIS) Tabellen durchzuführen. Sie müssen den zu verwendenden Datenbank Provider (siehe :doc:`db`) und die zu durchsuchende Spalte einer Tabelle angeben. Wenn Sie eine Suche über mehrere Spalten durchführen möchten, wird dies durch einen View gelöst. Die Tabellenkonfiguration ist der Tabellenname (optional mit Schema) und mindestens eine der beiden folgenden Spalten:

- ``searchColumn`` ist der Ort, an dem nach dem ``Schlüsselwort`` gesucht werden soll. Wenn nicht konfiguriert, wird das Schlüsselwort ignoriert.
- ``geometryColumn`` wird verwendet, um die Suche räumlich einzuschränken. Wenn nicht konfiguriert, wird der Parameter ``shape`` ignoriert.

WMS
~~~

Der Befehl ``GetFeatureInfo`` implementiert Anfragen an WMS-Quellen.
