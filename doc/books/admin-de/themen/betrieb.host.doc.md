# Host-Konfiguration :/admin-de/themen/betrieb/host

Bevor es um die Konfiguration der WebSuite selbst geht, steht die Umgebung, in der sie läuft. Dieser Abschnitt beschreibt, was auf dem Host vorhanden sein muss und wie die Teile zusammenhängen.

Die GBD WebSuite wird als Docker-Container ausgeliefert. Auf dem Host benötigen Sie daher eine Container-Runtime mit dem Compose-Plugin – mehr nicht: Alle Bibliotheken, der Webserver und der Kartenproxy stecken im Image. Eine Installation besteht in der Regel aus zwei Containern, der WebSuite und dem QGIS-Server, die sich dieselben Verzeichnisse teilen.

## Zwei Verzeichnisse

Alles, was Sie beisteuern, liegt in zwei Verzeichnissen, die in die Container eingebunden werden. Ihre Trennung ist die wichtigste Entscheidung beim Aufsetzen einer Installation:

- **`/data`** enthält alles, was Sie schreiben und versionieren: die Konfigurationsdateien, QGIS-Projekte, Vorlagen, statische Dateien. Die WebSuite liest dieses Verzeichnis nur; Sie können es schreibgeschützt einbinden.
- **`/gws-var`** enthält alles, was die WebSuite selbst erzeugt: Caches, Sitzungen, Zwischenergebnisse. Es muss beschreibbar sein und einen Neustart überdauern – sonst melden sich angemeldete Nutzer nach jedem Neustart erneut an, und der Kartencache baut sich von vorn auf.

Was den Platzbedarf treibt, ist fast immer `/gws-var`: Ein gefüllter [Kachel-Cache](/admin-de/themen/karten/cache) wächst schnell auf viele Gigabyte und verbraucht dabei eine große Zahl von Inodes.

## Netzwerk

Nach außen gibt die WebSuite die Ports für HTTP und HTTPS frei. Nach innen müssen sich die Container gegenseitig erreichen: Die WebSuite spricht den QGIS-Server über seinen Dienstnamen im Compose-Netz an.

Ein Sonderfall ist die Datenbank. Läuft PostgreSQL als weiterer Container oder auf einem entfernten Server, ist die Anbindung unproblematisch. Läuft sie dagegen unmittelbar auf dem Host, führt `localhost` ins Leere – innerhalb des Containers meint `localhost` den Container selbst. Dieser Fall braucht eine ausdrückliche [Adresse für den Host](/admin-de/konfiguration/host).

## Zugangsdaten

Datenbank-Zugangsdaten gehören nicht in die Konfiguration. Legen Sie sie in einer `pg_service.conf` ab und verweisen Sie in der Konfiguration nur auf den Dienstnamen. Das hält Passwörter aus versionierten Dateien heraus und hat einen zweiten Vorteil: QGIS-Projekte und die WebSuite greifen auf dieselben Verbindungen zu, sodass ein in QGIS erstelltes Projekt ohne Anpassung im Server funktioniert.

Damit das auch für den QGIS-Server gilt, muss `PGSERVICEFILE` in **beiden** Containern gesetzt sein.

%see
Siehe auch: [Konfiguration/Host-Konfiguration](/admin-de/konfiguration/host).
%end
