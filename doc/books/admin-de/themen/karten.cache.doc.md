# Caching :/admin-de/themen/karten/cache

Der Server kann Geo-Bilder aus externen Quellen auf der Festplatte zwischenspeichern (*cachen*), sodass wiederholte Anfragen deutlich schneller beantwortet werden. Der Cache liegt im veränderlichen Datenverzeichnis und kann bei Bedarf jederzeit vollständig gelöscht werden.

%warn
Caches können viel Speicherplatz belegen. Achten Sie auf ausreichend freien Platz und freie Inodes.
%end

Das Zwischenspeichern konfigurieren Sie pro Rasterlayer. Über `maxLevel` begrenzen Sie die tiefste (feinste) Zoomstufe, die zwischengespeichert wird (Vorgabe 1), über `maxAge`, wie lange ein Bild gültig bleibt, bevor es neu geholt wird (Vorgabe 7 Tage). Der Cache füllt sich von selbst, sobald Nutzer die Karten betrachten; er lässt sich aber auch vorab über die Kommandozeile befüllen (*Seeding*) und dort verwalten. Nach Änderungen an der Ansicht oder an einem Layer sollten Sie den betroffenen Cache löschen, um Darstellungsfehler zu vermeiden.

Das Seeding ist durch dieselben Grenzen bestimmt; global regeln `cache.seedingConcurrency` und `cache.seedingMaxTime` die Zahl der Threads und die maximale Laufzeit eines Durchlaufs.

%warn
Antwortet die Quelle beim Seeding mit einem Fehler, speichert der Server an dieser Stelle ein **leeres Bild**. Ein Ausfall der Quelle während des Vorbefüllens hinterlässt daher weiße Kacheln, die erst nach dem Löschen des Caches wieder korrekt geladen werden.
%end

%see
Siehe auch: [Konfiguration/Kommandozeilen-Befehle](/admin-de/konfiguration/cli).
%end
