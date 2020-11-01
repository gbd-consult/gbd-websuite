Suche
=====

Die GBD WebSuite implementiert eine einheitliche Suchfunktionalität, die mit verschiedenen Arten von Raum- und Sachdaten arbeitet. Grundsätzlich enthält eine Suchanfrage an den Server diese drei Parameter:

{TABLE}
    ``keyword`` | ein Stichwort, nach dem gesucht werden soll
    ``shape`` | eine Geometrie, falls vorhanden, ist die Suche räumlich auf diese Geometrie beschränkt.
    ``layers`` | eine Liste der Ebenen, auf denen die Suche durchgeführt werden soll. Das sind normalerweise die im Client sichtbare bzw selektierte Ebene
{/TABLE}

Es muss entweder ``keyword`` oder ``shape`` oder beides vorhanden sein.

Die von anderen GIS-Systemen bekannte "Identifizieren" Funktion fällt in GWS auch unter dem Begriff Suche, in dem Fall, eine ohne Stichwort und mit einer Punkt-Geometrie.

Wenn der GBD WebSuite Server eine Suchanfrage erhält, verbindet er sich mit den konfigurierten Suchanbietern (``provider``) und verwendet automatisch die für jede Quelle gültige Methode. Beispielsweise werden für WMS-Quellen "GetFeatureInfo"-Anfragen gestellt, für Datenbankdaten wird eine "SELECT"-Abfrage durchgeführt und so weiter. Sobald der Server Ergebnisse aus verschiedenen Quellen erhält, werden sie konsolidiert, optional transformiert und formatiert (s. ^feature) und als Liste einheitlicher Features an den Client zurückgegeben.

Die Anbieter werden für das ganze Projekt oder für individuelle Layers konfiguriert. Die Layer-Anbieter werden nur dann aktiviert, wenn der ``layers`` Liste den entsprechende Layer-ID enthält.

Allgemeine Optionen
-------------------

Für jeden Suchanbieter können Sie folgende allgemeine Optionen konfigurieren:

{TABLE}
    ``dataModel`` | Transformation-Regel für Features (s. ^feature)
    ``defaultContext`` |  räumliche Einschränkung der Suche, falls keine Geometrie vorhanden ist: die ganze Karte (``map``) oder die aktuelle Ansicht (``view``)
    ``templates`` | Formatierungsvorlagen für Features (s. ^feature)
    ``tolerance`` | räumliche Toleranz, entweder in Bildschirmpixeln (wie ``10px``) oder Karteneinheiten (wie ``10m``)
    ``withGeometry`` |  wenn das auf ``false`` gesetzt wird, werden räumliche Abfragen von diesem Anbieter nicht bearbeitet (reine Stichwort-Suche)
    ``withKeyword`` |  wenn das auf ``false`` gesetzt wird, werden Abfragen mit dem Stichwort von diesem Anbieter nicht bearbeitet (rein räumliche Suche)
{/TABLE}

Suchanbieter
------------

nominatim
~~~~~~~~~

^REF gws.ext.search.provider.nominatim.Config

Schnittstelle zu `Nominatim <https://nominatim.openstreetmap.org//>`_, der OSM-Suchmaschine. Sie können die Parameter ``country`` und ``language`` konfigurieren, um die Suchergebnisse anzupassen.

postgres
~~~~~~~~

^REF gws.ext.search.provider.postgres.Config

Bietet direkte Suche in Postgres/PostGIS Tabellen. Sie müssen und die zu durchsuchende Tabelle (``table``) angeben (s. ^db). Zusätzlich können Sie die Spalte wo nach dem Stichwort gesucht werden soll (`searchColumn``) konfigurieren.

Wenn der Server eine Abfrage mit ``keyword`` und ``shape`` bearbeitet, wird ungefähr folgende SQL Abfrage ausgeführt: ::

    SELECT * FROM {table}
        WHERE
            searchColumn ILIKE %{keyword}%
            AND ST_Intersects(geometryColumn, {shape})

Anders gesagt, wird Stichword case-insensitiv mit beliebiger Platzierung und Geometrie mit der Überschneidung gesucht. Wenn ``searchColumn`` bzw ``geometryColumn`` fehlen, wird in dieser Abfrage nur eine Bedingung stehen.

^NOTE In der Zukunft wird es möglich sein, diese Abfrage flexibel zu konfigurieren.

wfs
~~~

^REF gws.ext.search.provider.wfs.Config

Implementiert ``GetFeature``-Anfragen für WFS-Quellen. Sie müssen die Service-URL bereitstellen. Es ist auch möglich, die Suche auf bestimmte Layer (oder "Typen") zu beschränken.

wms
~~~

^REF gws.ext.search.provider.wms.Config

Implementiert ``GetFeatureInfo`` Anfragen für WMS-Quellen. Wie bei WFS, muss die Service-URL und optional die Suchlayers konfiguriert werden.

qgispostgres/qgiswms
~~~~~~~~~~~~~~~~~~~~

Diese Anbieter werden intern bei der Konfiguration von QGIS Projekten verwendet.

Aktion ``search``
-----------------

Mit dieser Aktion schalten Sie die Suchfunktionalität frei. Mit dem Parameter ``limit`` können Sie steuern, wie viele Suchergebnisse der Server zurückgibt (unter allen Anbietern).

^NOTE In der Zukunft können Sie auch ein ``limit`` pro Anbieter steuern.
