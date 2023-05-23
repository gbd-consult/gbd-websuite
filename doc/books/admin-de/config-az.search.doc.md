# Suche :/admin-de/config-az/search

%reference_de 'gws.ext.config.finder'

Die GBD WebSuite implementiert eine einheitliche Suchfunktionalität, die mit verschiedenen Arten von Raum- und Sachdaten arbeitet. Grundsätzlich enthält eine Suchanfrage an den Server diese drei Parameter:

| OPTION | BEDEUTUNG |
|---|---|
| *keyword* | ein Stichwort, nach dem gesucht werden soll |
| *shape* | eine Geometrie. Falls vorhanden, ist die Suche räumlich auf diese Geometrie beschränkt. |
| *layers* | eine Liste der Ebenen, auf denen die Suche durchgeführt werden soll. Das sind normalerweise die im Client sichtbare bzw selektierte Ebene |


Die von anderen GIS-Systemen bekannte "Identifizieren" Funktion fällt in GWS auch unter dem Begriff Suche, in dem Fall, eine ohne Stichwort und mit einer Punkt-Geometrie.

Wenn der GBD WebSuite Server eine Suchanfrage erhält, verbindet er sich mit den konfigurierten Suchanbietern (``provider``) und verwendet automatisch die für jede Quelle gültige Methode. Beispielsweise werden für WMS-Quellen "GetFeatureInfo"-Anfragen gestellt, für Datenbankdaten wird eine "SELECT"-Abfrage durchgeführt und so weiter. Sobald der Server Ergebnisse aus verschiedenen Quellen erhält, werden sie konsolidiert, optional transformiert und formatiert (siehe [Feature](/admin-de/config-az/feature)) und als Liste einheitlicher Features an den Client zurückgegeben.

Die Anbieter werden für das ganze Projekt oder für individuelle Layers konfiguriert. Die Layer-Anbieter werden nur dann aktiviert, wenn der ``layers`` Liste den entsprechende Layer-ID enthält.

Mit der Aktion ``search`` schalten Sie die Suchfunktionalität frei. Mit dem Parameter ``limit`` können Sie steuern, wie viele Suchergebnisse der Server zurückgibt (unter allen Anbietern).

%info
 In der Zukunft können Sie auch ein ``limit`` pro Anbieter steuern.
%end

## Allgemeine Optionen

Für jeden Suchanbieter können Sie folgende allgemeine Optionen konfigurieren:

| OPTION | BEDEUTUNG |
|---|---|
| ``dataModel`` | Transformation-Regel für Features, siehe [Feature](/admin-de/config-az/feature) |
| ``defaultContext`` |  räumliche Einschränkung der Suche, falls keine Geometrie vorhanden ist: die ganze Karte (``map``) oder die aktuelle Ansicht (``view``) |
| ``templates`` | Formatierungsvorlagen für Features, siehe [Feature](/admin-de/config-az/feature) |
| ``tolerance`` | räumliche Toleranz, entweder in Bildschirmpixeln (wie ``10px``) oder Karteneinheiten (wie ``10m``) |
| ``withGeometry`` |  wenn das auf ``false`` gesetzt wird, werden räumliche Abfragen von diesem Anbieter nicht bearbeitet (reine Stichwort-Suche) |
| ``withKeyword`` |  wenn das auf ``false`` gesetzt wird, werden Abfragen mit dem Stichwort von diesem Anbieter nicht bearbeitet (rein räumliche Suche) |

## Suchanbieter

### nominatim

TODO! %reference_de 'gws.ext.search.provider.nominatim.Config'

Schnittstelle zu [Nominatim](https://nominatim.openstreetmap.org/), der OSM-Suchmaschine. Sie können die Parameter ``country`` und ``language`` konfigurieren, um die Suchergebnisse anzupassen.

### postgres

%reference_de 'gws.plugin.postgres.finder.Config'

Bietet direkte Suche in Postgres/PostGIS Tabellen. Sie müssen und die zu durchsuchende Tabelle (``tableName``) angeben (siehe [Datenbanken](/admin-de/config-az/database)). Zusätzlich können Sie die Spalte in der nach dem Stichwort gesucht werden soll (``searchColumn``) konfigurieren.

Wenn der Server eine Abfrage mit *keyword* und *shape* bearbeitet, wird ungefähr folgende SQL Abfrage ausgeführt:

```sql

    SELECT * FROM {table}
        WHERE
            {searchColumn} ILIKE %{keyword}%
            AND ST_Intersects({geometryColumn}, {shape})
```

Anders gesagt, wird Stichword case-insensitiv mit beliebiger Platzierung und Geometrie mit der Überschneidung gesucht. Wenn ``searchColumn`` bzw ``geometryColumn`` fehlen, wird in dieser Abfrage nur eine Bedingung stehen.

%info
 In der Zukunft wird es möglich sein, diese Abfrage flexibel zu konfigurieren.
%end

### wfs

%reference_de 'gws.plugin.ows_provider.wfs.finder.Config'

Implementiert ``GetFeature``-Anfragen für WFS-Quellen. Sie müssen die Service-URL bereitstellen. Es ist auch möglich, die Suche auf bestimmte Layer (oder "Typen") zu beschränken.

### wms

%reference_de 'gws.plugin.ows_provider.wms.finder.Config'

Implementiert ``GetFeatureInfo`` Anfragen für WMS-Quellen. Wie bei WFS, muss die Service-URL und optional die Suchlayers konfiguriert werden.

### qgispostgres/qgiswms

TODO! noch gültig?

Diese Anbieter werden intern bei der Konfiguration von QGIS Projekten verwendet.
