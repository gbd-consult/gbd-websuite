# Suche :/admin-de/themen/suche

Die GBD WebSuite implementiert eine einheitliche Suchfunktionalität, die mit verschiedenen Arten von Raum- und Sachdaten arbeitet. Eine Suchanfrage an den Server enthält grundsätzlich drei Parameter:

| PARAMETER | BEDEUTUNG |
|---|---|
| *keyword* | ein Stichwort, nach dem gesucht werden soll |
| *shape* | eine Geometrie – falls vorhanden, ist die Suche räumlich auf diese Geometrie beschränkt |
| *layers* | eine Liste der Ebenen, auf denen die Suche durchgeführt werden soll (normalerweise die im Client sichtbaren bzw. selektierten Ebenen) |

Die von anderen GIS-Systemen bekannte „Identifizieren"-Funktion fällt in der GBD WebSuite ebenfalls unter den Begriff Suche – in diesem Fall ohne Stichwort und mit einer Punkt-Geometrie.

Wenn der Server eine Suchanfrage erhält, verbindet er sich mit den konfigurierten Suchanbietern (``provider``) und verwendet automatisch die für jede Quelle geeignete Methode: Für WMS-Quellen werden ``GetFeatureInfo``-Anfragen gestellt, für Datenbankdaten wird eine ``SELECT``-Abfrage durchgeführt usw. Die Ergebnisse aus verschiedenen Quellen werden konsolidiert, optional transformiert und formatiert und als Liste einheitlicher Features an den Client zurückgegeben.

### Suchfunktion freischalten :/admin-de/themen/suche/aktion

Die Suchfunktionalität wird über die Aktion ``search`` freigeschaltet. Mit dem Parameter ``limit`` steuern Sie, wie viele Suchergebnisse der Server insgesamt zurückgibt (über alle Anbieter hinweg):

```javascript
{
    actions+ {
        type search
        limit 100
    }
}
```

Suchanbieter werden entweder für das gesamte Projekt oder für individuelle Layer konfiguriert. Layer-Anbieter werden nur dann aktiviert, wenn die ``layers``-Liste die entsprechende Layer-ID enthält.

### Allgemeine Anbieteroptionen :/admin-de/themen/suche/optionen

Für jeden Suchanbieter stehen folgende allgemeine Optionen zur Verfügung:

| OPTION | BEDEUTUNG |
|---|---|
| ``dataModel`` | Transformationsregel für Features
| ``defaultContext`` | räumliche Einschränkung der Suche, falls keine Geometrie vorhanden ist: die ganze Karte (``map``) oder die aktuelle Ansicht (``view``) |
| ``templates`` | Formatierungsvorlagen für Features
| ``tolerance`` | räumliche Toleranz, entweder in Bildschirmpixeln (z. B. ``10px``) oder Karteneinheiten (z. B. ``10m``) |
| ``withGeometry`` | wenn auf ``false`` gesetzt, werden räumliche Abfragen von diesem Anbieter nicht bearbeitet (reine Stichwortsuche) |
| ``withKeyword`` | wenn auf ``false`` gesetzt, werden Abfragen mit Stichwort von diesem Anbieter nicht bearbeitet (rein räumliche Suche) |

### Nominatim :/admin-de/themen/suche/nominatim

%reference_de 'gws.ext.search.provider.nominatim.Config'

Nominatim ist die Suchmaschine von OpenStreetMap und bietet eine Schnittstelle zur Adress- und Ortssuche auf Basis der OSM-Daten. Die GBD WebSuite unterstützt die Anbindung an einen Nominatim-Dienst direkt über den Suchanbieter ``nominatim``.

Mit den Parametern ``country`` und ``language`` lassen sich die Suchergebnisse geografisch und sprachlich eingrenzen:

```javascript
{
    search.providers+ {
        type nominatim
        url "https://nominatim.openstreetmap.org"
        country "de"
        language "de"
    }
}
```

%info
Für den produktiven Einsatz empfiehlt sich eine eigene Nominatim-Instanz, da der öffentliche Dienst unter nominatim.openstreetmap.org Nutzungsbeschränkungen unterliegt.
%end

### Postgres :/admin-de/themen/suche/postgres

%reference_de 'gws.plugin.postgres.finder.Config'

Der Postgres-Suchanbieter ermöglicht die direkte Suche in PostgreSQL/PostGIS-Tabellen. Neben der Datenbankverbindung (siehe [PostgreSQL](/admin-de/themen/postgresql)) wird mindestens die zu durchsuchende Tabelle (``tableName``) benötigt. Optional kann die Spalte für die Stichwortsuche (``searchColumn``) angegeben werden.

```javascript
{
    search.providers+ {
        type postgres
        db "meine_datenbank"
        tableName "adressen"
        searchColumn "bezeichnung"
        geometryColumn "geom"
    }
}
```

Wenn der Server eine Abfrage mit *keyword* und *shape* bearbeitet, wird ungefähr folgende SQL-Abfrage ausgeführt:

```sql
SELECT * FROM {table}
    WHERE
        {searchColumn} ILIKE %{keyword}%
        AND ST_Intersects({geometryColumn}, {shape})
```

Das Stichwort wird dabei case-insensitiv mit beliebiger Platzierung gesucht; die Geometrie wird über Überschneidung geprüft. Fehlt ``searchColumn`` oder ``geometryColumn``, entfällt die jeweilige Bedingung in der Abfrage.

### WFS :/admin-de/themen/suche/wfs

%reference_de 'gws.plugin.ows_provider.wfs.finder.Config'

Der WFS-Suchanbieter implementiert ``GetFeature``-Anfragen für WFS-Quellen. Es muss die Service-URL angegeben werden. Optional kann die Suche auf bestimmte Layer (in WFS-Terminologie „Typen") eingeschränkt werden:

```javascript
{
    search.providers+ {
        type wfs
        url "https://example.com/wfs"
        layers ["strassen", "gebaeude"]
    }
}
```

### WMS :/admin-de/themen/suche/wms

%reference_de 'gws.plugin.ows_provider.wms.finder.Config'

Der WMS-Suchanbieter implementiert ``GetFeatureInfo``-Anfragen für WMS-Quellen. Wie beim WFS muss die Service-URL angegeben werden; optional können die zu befragenden Suchlayer konfiguriert werden:

```javascript
{
    search.providers+ {
        type wms
        url "https://example.com/wms"
        layers ["orthofotos", "flurstuecke"]
    }
}
```
