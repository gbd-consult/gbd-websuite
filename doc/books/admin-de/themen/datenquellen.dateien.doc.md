# Dateien :/admin-de/themen/datenquellen/dateien

Liegen die Daten als Datei auf dem Server, binden Sie sie unmittelbar ein – ohne Datenbank und ohne vorgeschalteten Dienst. Die Datei muss dafür im Container erreichbar sein; die Verzeichnisse dafür richten Sie in der [Host-Konfiguration](/admin-de/themen/betrieb/host) ein.

Für Vektordaten ist GeoJSON das Dateiformat der Wahl. Der Layer-Typ `geojson` stellt die Features einer Datei dar, deren Pfad Sie im `provider` unter `path` angeben. Dieselbe Quelle steht auch als Modell und als Finder zur Verfügung, sodass Sie die Attribute beschreiben und die Datei durchsuchen können.

Rasterdaten kommen auf zwei Wegen. Der Layer-Typ `raster` liest Rasterdateien über GDAL – etwa GeoTIFF – und nimmt entweder eine Liste von Pfaden (`paths`) oder ein Suchmuster (`pathPattern`) entgegen; lässt sich das Koordinatensystem nicht aus der Datei ermitteln, geben Sie es im `provider` mit `crs` an. Für die Gestaltung hinterlegen Sie mit `sldPath` eine SLD-Datei. Der Layer-Typ `mbtiles` liest eine MBTiles-Datei, also bereits fertig gekachelte Rasterdaten in einer SQLite-Datenbank, und benötigt nur deren `path`. Beide Typen kennen `transparentColor` für eine als durchsichtig zu behandelnde Farbe und `processing` für weitere Verarbeitungsdirektiven.

Ein QGIS-Projekt ist ebenfalls eine Datei, liefert aber keine einzelne Quelle, sondern einen ganzen Layerbaum – dafür gibt es einen eigenen Abschnitt.

%see
Siehe auch: [Layer/geojson](/admin-de/konfiguration/layer/geojson), [Layer/raster](/admin-de/konfiguration/layer/raster), [Layer/mbtiles](/admin-de/konfiguration/layer/mbtiles), [Modell/geojson](/admin-de/konfiguration/model/geojson), [Finder/geojson](/admin-de/konfiguration/finder/geojson).
%end
