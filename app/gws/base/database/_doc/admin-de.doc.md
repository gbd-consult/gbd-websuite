# Datenbanken :/admin-de/config/db

Die GBD WebSuite kann Geodaten aus Datenbanken beziehen, entweder indirekt über QGIS und WMS/WMTS-Anbieter oder durch direkte Anbindung an eine Datenbank. Im letzteren Fall müssen Sie die Datenbankverbindungen und Tabellen konfigurieren.

## Datenbank-Anbieter :/admin-de/config/db/type/

Eine Anbieter (``prodiver``) Konfiguration beschreibt, welche Datenbanken und mit welchen Zugangsdaten verwendet werden. Zur Zeit unterstützen wir nur PostgreSQL/PostGIS Datenbanken.

%info
 In der Zukunft sind auch Sqlite/SpatiaLite, MySQL und MongoDB geplant.
%end

### :/admin-de/config/db/type/postgres

## Zugriffsrechte

Wir empfehlen Ihrem Datenbank-Nutzer möglichst wenige Rechte zu vergeben. Für die Funktionen wie [Suche](/admin-de/config/suche) oder [Postgres-Layer](/admin-de/config/layer) ist ein ``SELECT`` ausreichend, für Editierfunktionen ([edit](/admin-de/plugin/edit) oder [tabedit](/admin-de/plugin/tabedit)) braucht man auch ``INSERT`` und ``UPDATE``. Wenn Sie ALKIS [Alkis](/admin-de/plugin/alkis) verwenden, muss der DB-Nutzer auch ``CREATE`` und ``DROP`` für das GWS-Arbeitsschema besitzen.

## Datenbank-Tabellen

%reference_de 'gws.plugin.postgres.model.Config'

Bei einigen GBD WebSuite Funktionen wie z.B. [Suche](/admin-de/config/suche) oder [Digitalisierung](/admin-de/plugin/edit) ist eine Tabellen-Konfiguration notwendig. Minimal ist ein Tabellen-Namen anzugeben (optional mit einem Schema). Sie können auch die Namen für Primärschlüssel (``keyColumn``) und Geometrie-Spalte (``geometryColumn``) angeben, per Default versucht das System diese Werte aus ``INFORMATION_SCHEMA`` und ``GEOMETRY_COLUMNS`` automatisch zu ermitteln.

Falls Sie mehrere Anbieter verwenden, müssen Sie auch die Anbieter ``uid`` in der Tabellen-Konfiguration angeben.
