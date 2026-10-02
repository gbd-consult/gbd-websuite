# Datenbanken :/admin-de/themen/datenquellen/datenbanken

Als Datenbank unterstützt die GBD WebSuite PostgreSQL mit der Erweiterung PostGIS. Wie Sie die Verbindung als Provider einrichten und die Zugangsdaten ablegen, behandelt das Thema [](/admin-de/themen/daten/datenbanken).

Der Layer-Typ `postgres` stellt die Geometrien einer Tabelle als Vektorlayer dar. Die Tabelle geben Sie mit `tableName` an, bei mehreren konfigurierten Verbindungen wählen Sie mit `dbUid` die passende aus. Geometrietyp und Koordinatensystem ermittelt die WebSuite aus der Tabelle selbst, Sie müssen sie nicht wiederholen.

Die Datenbank ist die einzige Quelle, die auch schreibend genutzt werden kann: Über das Modell `postgres` und die Aktion `edit` legen Nutzer Objekte an, ändern und löschen sie. Alle anderen Quellen sind ausschließlich lesend. Wie Sie das [](/admin-de/themen/daten/editieren) einrichten und wer es darf, beschreibt ein eigenes Thema.

%see
Siehe auch: [Layer/postgres](/admin-de/konfiguration/layer/postgres), [Modell/postgres](/admin-de/konfiguration/model/postgres), [Finder/postgres](/admin-de/konfiguration/finder/postgres).
%end
