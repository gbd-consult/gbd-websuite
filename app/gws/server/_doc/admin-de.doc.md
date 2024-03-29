# Server Konfiguration :/admin-de/config/server

%reference_de 'gws.server.core.Config'

## Loggen

Das Loggen erfolgt per Default in Standard-Ausgabe (*stdout*), sodass Logs mit ``docker logs`` bzw ``--log-driver`` Option abgefangen werden können (s. https://docs.docker.com/config/containers/logging/ für mehr Info). Alternativ können Sie GWS Logs in eine Datei weiterleiten.

Für das Loggen sind folgende Stufen definiert:


| TAG | EIGENSCHAFT |
|---|---|
| ``ERROR`` | schwerwiegende Fehler, die die Funktion des GWS Servers beeinträchtigen |
| ``INFO`` | zusätzliche Informationsmeldungen (Defaultwert) |
| ``DEBUG`` | zusätzliche Ausgaben zur Fehlersuche |

## Module

Die GBD WebSuite betreibt intern mehrere Servermodule:

- das ``web``-Modul, das eingehende Anfragen entgegennimmt und versendet [Web Server](/admin-de/config/web)
- das ``mapproxy`` Modul, das den gebündelten MapProxy ausführt und sich um externe Quellen, Caching und Umprojizieren kümmert
- das ``qgis`` Modul, das den gebündelten QGIS Server betreibt und QGIS Projekte und Layer rendert [QGIS](/admin-de/intro/concepts)
- das ``spool`` Modul, das den Druck und andere Hintergrundaufgaben übernimmt.
- das ``monitor`` Modul, das das Dateisystem überwacht, und bei den Änderungen in Ihren Daten einen Hot-Reload des Servers durchführt

Jedes Modul kann mit ``enabled: false`` deaktiviert werden, wenn es nicht benötigt wird (z.B. wenn Sie keine QGIS-Projekte verwenden, brauchen Sie den QGIS-Server nicht auszuführen). Sie können auch die Anzahl der *Prozessen* (``worker``) konfigurieren, die jedes Modul verwenden darf.

Für Hochlast-Workflows ist es auch möglich, verschiedene Module auf verschiedenen physikalischen Maschinen zu betreiben. Beispielsweise können Sie eine GWS-Installation einrichten, die nur das Mapproxy-Modul, eine weitere für den QGIS-Server und eine weitere für das Frontend-Web-Modul ausführt. In diesem Fall können Sie für Mapproxy und QGIS in der Web-Konfiguration ``host`` and ``port`` angeben, so dass diese über das Netzwerk abgefragt werden können.

## Monitor

Normalerweise, überwacht der Montior folgende Daten und Ordner:

- App-Konfigurationsdatei
- Projekt-Konfigurationsdateien
- Projekt-Konfigurations-Ordner, falls Sie die Option ``projectDirs`` verwenden [Applikation](/admin-de/config/applikation)
- Vorlagen
- QGIS Projekte

In der Konfiguration des Monitors können Sie bestimmte Pfade mit der ``ignore`` Option ausschließen.
