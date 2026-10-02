# Kommandozeile :/admin-de/themen/betrieb/kommandozeile

Verwaltungsaufgaben erledigen Sie über das Kommandozeilenwerkzeug `gws`, das im laufenden Container ausgeführt wird:

    docker exec -it <container_name> gws -h

Der Aufruf ohne Argumente oder mit `-h` listet die verfügbaren Befehle; `gws <gruppe> -h` zeigt die Hilfe zu einer einzelnen Gruppe. Welche Befehle es gibt und was sie im Einzelnen tun, führen die [](/admin-de/konfiguration/cli) auf.

## Jeder Befehl besteht aus zwei Wörtern

Ein Aufruf hat immer die Form `gws <gruppe> <verb>`, gefolgt von benannten Optionen:

    gws auth password --user meier
    gws cache status
    gws server reconfigure

Beide Wörter sind zwingend — ein Aufruf mit nur einem Wort bricht mit `invalid arguments` ab. Intern setzt die WebSuite die beiden zu einem Namen in Binnenmajuskel zusammen, aus `gws auth password` wird `authPassword`. Diesen zusammengesetzten Namen sehen Sie in Protokollen und Fehlermeldungen; auf der Kommandozeile geben Sie ihn **nicht** so ein.

Optionen schreiben Sie mit zwei Bindestrichen und in Binnenmajuskel, etwa `--projectUid`. Groß- und Kleinschreibung spielt dabei keine Rolle.

## Die Befehle arbeiten gegen Ihre Konfiguration

`gws` lädt beim Start dieselbe Konfiguration wie der Server. Ein Befehl kann deshalb an einem Konfigurationsfehler scheitern, der mit seiner eigentlichen Aufgabe nichts zu tun hat. Vor jedem Eingriff in den laufenden Betrieb lohnt daher:

    gws server configtest

Dasselbe gilt für Plugins. Welche Plugins geladen werden, steht in der Manifest-Datei, und `gws` sucht sie an drei Stellen in dieser Reihenfolge: in der Option `--manifest`, in der Umgebungsvariablen `GWS_MANIFEST`, und schließlich unter `/data/MANIFEST.json`, sofern diese Datei existiert.

%warn
Die letzte Stufe — der Rückfall auf `/data/MANIFEST.json` — gilt nur für `gws` und für den Server, **nicht** für `make.sh`. Die Bauskripte kennen ausschließlich `--manifest` und `GWS_MANIFEST`. Ein Bau ohne diese Angabe übergeht Ihre Plugins stillschweigend und ohne Fehlermeldung, während der Server sie danach weiterhin lädt. Setzen Sie `GWS_MANIFEST` im `environment`-Block des Containers, dann gilt die Angabe für beides.
%end

## Berechtigungen

Befehle laufen als Systembenutzer und umgehen damit die Zugriffsregeln, die im Betrieb greifen. Was sich auf der Kommandozeile erzeugen oder abfragen lässt, muss für einen angemeldeten Nutzer noch lange nicht verfügbar sein. Prüfen Sie Berechtigungsfragen deshalb nie mit `gws`, sondern immer mit einer echten Anmeldung.

%see
Siehe auch: [Konfiguration/Kommandozeilen-Befehle](/admin-de/konfiguration/cli).
%end
