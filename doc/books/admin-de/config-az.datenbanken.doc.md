# Datenbanken :/admin-de/config-az/datenbanken

Die GBD WebSuite kann Geodaten aus Datenbanken beziehen, entweder indirekt über QGIS und WMS/WMTS-Anbieter oder durch direkte Anbindung an eine Datenbank. Im letzteren Fall müssen Sie die Datenbankverbindungen und Tabellen konfigurieren.

## Datenbank-Anbieter

Eine Anbieter (``prodiver``) Konfiguration beschreibt, welche Datenbanken und mit welchen Zugangsdaten verwendet werden. Zur Zeit unterstützen wir nur PostgreSQL/PostGIS Datenbanken.

%info
 In der Zukunft sind auch Sqlite/SpatiaLite, MySQL und MongoDB geplant.
%end

### postgres

%reference_de 'gws.plugin.postgres.provider.Config'

Wir unterstützen PostgreSQL ab Version 10 mit der obligatorisch PostGIS (ab 2.4) Extension.

Beispiel einer Postgres-Provider-Konfiguration:

```javascript
"database": {
    "providers" [
        {
            "database": "mydatabase",
            "type": "postgres",
            "uid": "my_db",
            "host": "172.17.0.1",
            "port": 5432,
            "username": "me",
            "password": "secret"
        }
    ]
}
```

Wenn Sie mehrere Server bzw. mehrere Zugangsdaten auf dem gleichen Server haben, müssen Sie diese als verschiedene Anbieter konfigurieren.

%info
 Auch wenn Ihr DB-Server sich auf demselben physischen Host befindet, können Sie nicht ``localhost`` als Hostname verwenden, weil die GBD WebSuite in einem Docker-Container läuft. Stattdessen sollte die IP-Adresse des Docker-Hosts wie ``172.17.0.1`` verwendet werden (der genaue Wert hängt von den Einstellungen Ihres Docker-Netzwerks ab). Aus Gründen der Portabilität ist es empfehlenswert, es mit ``--add-host`` zu aliasieren.
%end

## Zugriffsrechte

Wir empfehlen Ihrem Datenbank-Nutzer möglichst wenige Rechte zu vergeben. Für die Funktionen wie [Suche](/admin-de/config-az/suche) oder [Postgres-Layer](/admin-de/config-az/layer) ist ein ``SELECT`` ausreichend, für Editierfunktionen ([edit](/admin-de/plugin/digitalisierung) oder [tabedit](/admin-de/plugin/tabedit)) braucht man auch ``INSERT`` und ``UPDATE``. Wenn Sie ALKIS [Alkis](/admin-de/plugin/alkis) verwenden, muss der DB-Nutzer auch ``CREATE`` und ``DROP`` für das GWS-Arbeitsschema besitzen.

## Datenbank-Tabellen

%reference_de 'gws.plugin.postgres.model.Config'

Bei einigen GBD WebSuite Funktionen wie z.B. [Suche](/admin-de/config-az/suche) oder [Digitalisierung](/admin-de/plugin/digitalisierung) ist eine Tabellen-Konfiguration notwendig. Minimal ist ein Tabellen-Namen anzugeben (optional mit einem Schema). Sie können auch die Namen für Primärschlüssel (``keyColumn``) und Geometrie-Spalte (``geometryColumn``) angeben, per Default versucht das System diese Werte aus ``INFORMATION_SCHEMA`` und ``GEOMETRY_COLUMNS`` automatisch zu ermitteln.

Falls Sie mehrere Anbieter verwenden, müssen Sie auch die Anbieter ``uid`` in der Tabellen-Konfiguration angeben.
