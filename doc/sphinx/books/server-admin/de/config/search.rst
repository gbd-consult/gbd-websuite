Suche
=====

Die GBD WebSuite implementiert eine einheitliche Suchfunktionalität, die mit verschiedenen Arten von Raum- und Domänendaten arbeitet. Grundsätzlich enthält eine Suchanfrage an den Server diese drei Parameter:

TABLE
    *Stichwort* ~ Stichwort, nach dem gesucht werden soll
    *shape* ~ eine Geometrie, falls vorhanden, ist die Suche räumlich auf diese Geometrie beschränkt.
    *Layers* ~ eine Liste der Ebenen, auf denen die Suche durchgeführt werden soll
/TABLE

Wenn der GBD WebSuite Server eine Suchanfrage erhält, verbindet er sich mit den konfigurierten Datenquellen und verwendet automatisch die für jede Quelle gültige Methode (oder *Provider*). Beispielsweise werden für WMS-Quellen "GetFeatureInfo"-Anfragen gestellt, für Datenbankdaten wird eine "SELECT"-Abfrage durchgeführt und so weiter. Sobald der Server Ergebnisse aus verschiedenen Quellen erhält, werden sie konsolidiert, optional neu formatiert (s. ^features) und als Liste einheitlicher GeoJSON-Features an den Client zurückgegeben.

Anbieter suchen
---------------

Nominatim
~~~~~~~~~

Schnittstelle zu `Nominatim <https://nominatim.openstreetmap.org//>`_, der OSM-Suchmaschine. Sie können die Parameter ``Land`` und ``Sprache`` konfigurieren, um die Suchergebnisse anzupassen.

SQL
~~~

Bietet direkte Suche in SQL (PostGIS) Tabellen. Sie müssen den zu verwendenden DB to Provider (s. ^db) und die zu durchsuchende Tabelle angeben. Die Tabellenkonfiguration ist der Tabellenname (optional mit Schema) und mindestens eine der beiden folgenden Spalten:

- ``searchColumn`` ist der Ort, an dem nach dem ``Schlüsselwort`` gesucht werden soll. Wenn nicht konfiguriert, wird das Schlüsselwort ignoriert.
- ``geometryColumn`` wird verwendet, um die Suche räumlich einzuschränken. Wenn nicht konfiguriert, wird der Parameter ``shape`` ignoriert.

WMS
~~~

Implementiert ``GetFeatureInfo`` Anfragen für WMS-Quellen.

WFS
~~~

Implementiert ``GetFeatureInfo``-Anfragen für WFS-Quellen. Sie müssen die WFS-Service-URL bereitstellen. Es ist auch möglich, die Suche auf bestimmte Layer (oder "Typen") zu beschränken.
