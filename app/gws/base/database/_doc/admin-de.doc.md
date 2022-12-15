# Datenbanken :/admin-de/config/db

Die GBD WebSuite kann Geodaten aus Datenbanken beziehen, entweder indirekt über QGIS und WMS/WMTS-Anbieter oder durch direkte Anbindung an eine Datenbank. Im letzteren Fall müssen Sie die Datenbankverbindungen und Tabellen konfigurieren.

## Datenbank-Anbieter

Eine Anbieter (`prodiver`) Konfiguration beschreibt, welche Datenbanken und mit welchen Zugangsdaten verwendet werden. Zur Zeit unterstützen wir nur PostgreSQL/PostGIS Datenbanken.

^NOTE In der Zukunft sind auch Sqlite/SpatiaLite, MySQL und MongoDB geplant.

### :/admin-de/config/db/type/*

Zugriffsrechte
--------------

Wir empfehlen Ihrem Datenbank-Nutzer möglichst wenige Rechte zu vergeben. Für die Funktionen wie ^search oder Postgres-Layer (s. ^layer) ist ein `SELECT` ausreichend, für Editierfunktionen (^digitize oder ^tabedit) braucht man auch `INSERT` und `UPDATE`. Wenn Sie ALKIS (s. ^alkis) verwenden, muss der DB-Nutzer auch `CREATE` und `DROP` für das GWS-Arbeitsschema besitzen.

Datenbank-Tabellen
------------------

^REF gws.base.db.SqlTableConfig

Bei einigen GBD WebSuite Funktionen wie z.B. ^search oder ^digitize ist eine Tabellen-Konfiguration notwendig. Minimal ist ein Tabellen-Namen anzugeben (optional mit einem Schema). Sie können auch die Namen für Primärschlüssel (`keyColumn`) und Geometrie-Spalte (`geometryColumn`) angeben, per Default versucht das System diese Werte aus `INFORMATION_SCHEMA` und `GEOMETRY_COLUMNS` automatisch zu ermitteln.

Falls Sie mehrere Anbieter verwenden, müssen Sie auch die Anbieter `uid` in der Tabellen-Konfiguration angeben.



