# Koordinatensysteme :/admin-de/themen/karten/crs

Jede Karte wird in einem Koordinatenreferenzsystem dargestellt, das über einen EPSG-Code angegeben wird. Voreingestellt ist Web-Mercator, wie ihn die meisten Kacheldienste verwenden; in Deutschland kommen häufig die UTM-Systeme oder WGS 84 zum Einsatz.

Alle Layer eines Projekts werden in diesem System dargestellt. Stammen die Quelldaten aus einer anderen Projektion, projiziert die WebSuite sie automatisch um. Pro Projekt ist derzeit genau ein Koordinatensystem möglich.

Sie setzen es mit `crs` in der Karte des Projekts; ohne Angabe gilt `EPSG:3857`.

%see
Siehe auch: [Konfiguration/Karten](/admin-de/konfiguration/map).
%end
