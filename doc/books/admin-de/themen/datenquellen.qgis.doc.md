# QGIS-Projekte :/admin-de/themen/datenquellen/qgis

Wofür sich ein QGIS-Projekt einsetzen lässt, behandelt das Kapitel [](/admin-de/themen/qgis). Als Datenquelle betrachtet ist es keine einzelne Quelle: Die Daten hinter seinen Layern – Dateien, Datenbanktabellen, externe Dienste – bleiben Sache von QGIS, die WebSuite spricht das Projekt als Ganzes an.

Das Projekt selbst liegt entweder als Datei vor, deren Pfad Sie im `provider` unter `path` angeben, oder in einer PostgreSQL-Datenbank. Für die Ablage in der Datenbank benennen Sie mit `dbUid` die Verbindung sowie mit `schema` und `projectName` den Ort des Projekts darin. Damit lassen sich Projekte zentral pflegen, ohne Dateien auf den Server zu kopieren.

Standardmäßig lässt die WebSuite den QGIS-Server rendern und antworten. Für einzelne Datenquellen des Projekts können Sie das umgehen: `directRender` zählt die QGIS-Datenprovider auf, die die WebSuite selbst rendern soll, `directSearch` jene, die sie selbst durchsuchen soll. Das spart den Umweg über den QGIS-Server und ist vor allem bei PostGIS-Quellen spürbar.

%see
Siehe auch: [Layer/qgis](/admin-de/konfiguration/layer/qgis), [Layer/qgisflat](/admin-de/konfiguration/layer/qgisflat).
%end
