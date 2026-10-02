# Server & Prozesse :/admin-de/themen/betrieb/server

Der GBD WebSuite Server ist kein einzelner Prozess, sondern ein Verbund mehrerer Prozesse, die zusammenarbeiten:

- **nginx** – der Webserver, der Anfragen entgegennimmt und Antworten ausliefert.
- **Applikationsserver (uWSGI)** – führt die eigentliche Logik der WebSuite aus und bearbeitet die Anfragen.
- **Spooler** – erledigt langlaufende Aufgaben wie den Druck im Hintergrund.
- **Monitor** – überwacht die Konfigurations- und Datendateien und veranlasst bei Änderungen ein automatisches Neuladen.

Einzelne Prozesse lassen sich deaktivieren, wenn sie nicht benötigt werden.

Sie müssen auch nicht alle auf derselben Maschine laufen. Für den QGIS-Server geben Sie in der Konfiguration `host` und `port` an; damit lässt er sich auf einen eigenen Rechner auslagern, etwa um das Rendern von der Auslieferung zu trennen. Voreingestellt ist der lokale Betrieb.

## Dimensionierung

Unter Last bestimmen einige Angaben je Prozess das Verhalten des Servers. Jeder Prozess (`web`, `spool`) hat eine Anzahl von Arbeitsprozessen `workers` (Vorgabe 4). Für den Webserver begrenzen `timeout` (Vorgabe 60 s) und `maxRequestLength` (in MB) die Dauer und Größe einer Anfrage; für den Spooler bestimmen `jobFrequency`, wie oft nach neuen Aufgaben gesehen wird, und `timeout` die maximale Laufzeit einer Aufgabe.

## Automatisches Neuladen

Der Monitor prüft in Abständen (`frequency`, Vorgabe 30 s), ob sich Konfigurations- oder Datendateien geändert haben, und lädt bei Bedarf neu. Zwei Wege sind zu unterscheiden: Ein *Reload* startet die Arbeitsprozesse neu, ein *Reconfigure* liest die Konfiguration erneut ein; beide lassen sich auch über die Kommandozeile auslösen. Mit `disableWatch` schalten Sie die Überwachung ab, sodass nur noch ein ausdrücklicher Aufruf neu lädt.

Vor dem Konfigurieren und nach dem Start des Dienstes kann der Server eigene Skripte ausführen (`preConfigure`, `postConfigure`) – ein Einschubpunkt für Aufgaben, die zur Inbetriebnahme gehören.

## Konfigurationsvorlagen

Die Dokumente, die der Server für seine Dienste ausliefert, entstehen aus Vorlagen. Diese Vorlagen sind vorbelegt und lassen sich durch eigene ersetzen, um Aufbau und Inhalt der Ausgabe anzupassen.

Betroffen sind vor allem die XML-Dokumente der OWS-Dienste – etwa die Capabilities eines WMS oder die FeatureInfo-Antworten –, aber auch die Konfigurationsdateien, die der Server für seine internen Prozesse erzeugt. In den meisten Installationen genügen die mitgelieferten Vorlagen; eigene brauchen Sie erst, wenn die Ausgabe einer besonderen Vorgabe folgen muss, etwa einer INSPIRE-konformen Struktur.

Die Konfiguration der eingebetteten Dienste entsteht bei jedem Konfigurationslauf neu. Unter `server.templates` können Sie die zugrunde liegenden Vorlagen über ihr *Subject* ersetzen; vier sind definiert:

| Subject | erzeugt |
|---|---|
| `server.nginx_config` | `nginx.conf` für den vorgeschalteten Webserver |
| `server.uwsgi_config` | die Konfiguration je uWSGI-Prozess – `uwsgi_web.ini`, `uwsgi_spool.ini` |
| `server.rsyslog_config` | `syslog.conf` für den eingebetteten `rsyslogd`, nur im Container |
| `server.start_script` | das Startskript des Servers |

`server.uwsgi_config` wird dabei mehrfach ausgewertet, einmal je Prozess; welcher gemeint ist, steht der Vorlage im Argument `uwsgi` zur Verfügung.

%warn
Die erzeugten Dateien liegen unter `<GWS_VAR_DIR>/server/` und werden bei jedem Konfigurationslauf überschrieben. Eine dort von Hand geänderte `nginx.conf` ist beim nächsten `gws server reconfigure` wieder weg — Änderungen gehören in die Vorlage, nicht in das Ergebnis.
%end

Zum Einstieg in eine eigene Vorlage lohnt ein Blick in die mitgelieferte: Sie zeigt, welche Argumente zur Verfügung stehen und wie die erzeugte Datei aufgebaut ist.

- [Vorlage `nginx_config.cx.txt` im Quellcode](https://github.com/gbd-consult/gbd-websuite/blob/master/app/gws/server/templates/nginx_config.cx.txt)
- [API-Dokumentation `gws.server`](https://docs.gbd-websuite.de/stable/api/py/gws/server/index.html)
- [API-Dokumentation `gws.server.manager`](https://docs.gbd-websuite.de/stable/api/py/gws/server/manager/index.html) – beschreibt die Subjects und das Argument-Objekt der Vorlagen

## Logging

Der Server schreibt seine Meldungen nach *stdout*, sodass sie über die Log-Mechanismen des Containers abgerufen werden können. Die Ausführlichkeit steuern Sie über die Log-Stufe – von schwerwiegenden Fehlern bis zu ausführlichen Meldungen zur Fehlersuche.

%ref "gws.server.core.Config"
