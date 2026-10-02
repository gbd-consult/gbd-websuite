# Datenbanken :/admin-de/themen/daten/datenbanken

Die GBD WebSuite kann Geodaten direkt aus einer Datenbank beziehen; unterstützt wird PostgreSQL/PostGIS. Eine Datenbankverbindung wird als *Provider* konfiguriert und über ihre uid von Layern, Suchen und Modellen genutzt.

Ein Provider kann die Zugangsdaten unmittelbar enthalten. Empfohlen wird jedoch, die Verbindungen in einer `pg_service.conf` zu hinterlegen und in der Konfiguration nur den Dienstnamen anzugeben. So bleiben die Zugangsdaten außerhalb der Konfiguration und lassen sich zugleich von QGIS und der WebSuite gemeinsam nutzen.

%see
Siehe auch: [Konfiguration/Datenbank-Provider](/admin-de/konfiguration/databaseProvider).
%end
