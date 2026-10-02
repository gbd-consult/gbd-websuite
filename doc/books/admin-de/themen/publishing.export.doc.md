# Datenexport :/admin-de/themen/publishing/export

Die GBD WebSuite kann Features zum Herunterladen bereitstellen und dabei in unterschiedliche Ausgabeformate umwandeln. Ein *Exporter* beschreibt ein solches Format.

Exporter lassen sich global oder je Projekt konfigurieren. Global konfigurierte Exporter stehen allen Projekten zur Verfügung; ein im Projekt konfigurierter Exporter gilt nur für dieses Projekt. Ausgelöst wird der Export über die zuständige Server-Aktion.

## Gemeinsame Eigenschaften

Alle Exporter nutzen intern GDAL. Der jeweilige Typ (`csv`, `geojson`, `gml`, `kml`, `shapefile`) legt nur den GDAL-Treiber fest; die eigentlichen Treiber-Optionen geben Sie über das `options`-Objekt an – hier stellen Sie etwa die Zeichenkodierung eines Shapefiles oder das `srsName` einer GML-Datei ein. Ein separater Zeichensatz oder eine Umprojektion findet **nicht** statt: Der Export übernimmt die Koordinaten in ihrem vorhandenen System (einzelne Treiber wie GeoJSON und KML schreiben allerdings immer in WGS 84).

Beim Zusammenstellen der Features geht der Exporter tolerant vor und **überspringt** stillschweigend, was nicht in eine Tabelle passt: Features ohne Geometrie, mit abweichendem Geometrietyp oder abweichendem Koordinatensystem sowie Attribute mit nicht unterstützten Datentypen. Über `withNoGeometry`, `withMixedGeometry` und `withMixedCrs` lassen Sie diese Fälle ausdrücklich zu. Entstehen mehrere Dateien (etwa die Bestandteile eines Shapefiles), werden sie automatisch zu einem ZIP-Archiv zusammengefasst.

%see
Siehe auch: [Konfiguration/Exporter](/admin-de/konfiguration/exporter), [Aktion/exporter](/admin-de/konfiguration/action/exporter).
%end
