Server Konfiguration
====================

Die GBD WebSuite betreibt intern mehrere Servermodule:

- das ``web``-Modul, dass eingehende Anfragen entgegennimmt und versendet
- das ``mapproxy`` Modul, dass den gebündelten MapProxy ausführt und sich um externe Quellen, Caching und Reprojektionen kümmert
- das ``qgis`` Modul, dass den gebündelten QGIS Server betreibt und QGIS Projekte und Layer rendert
- das ``spool`` Modul, dass den Druck und andere Hintergrundaufgaben übernimmt.

Jedes Modul kann deaktiviert werden, wenn es nicht benötigt wird (z. B. wenn Sie keine QGIS-Projekte verwenden, brauchen Sie den QGIS-Server nicht auszuführen). Sie können auch die Anzahl der Arbeiter, CPU-Kerne und Threads konfigurieren, welche das jeweilige Modul verwenden darf. Standardmäßig stehen die Werte auf ``4`` und ``0``. Die optimalen Werte hängen jedoch von der Konfiguration Ihres Zielsystems ab.

Für sehr leistungsfordernde Workflows ist es auch möglich, verschiedene Module auf verschiedenen physikalischen Maschinen zu betreiben. Beispielsweise können Sie eine GBD WebSuite-Installation einrichten, der nur das Mapproxy-Modul zur Verfügung gestellt wird, eine weitere für den QGIS-Server und eine weitere damit das Frontend-Web-Modul ausgeführt wird. In diesem Fall können Sie für Mapproxy und QGIS in der Web-Konfiguration ``host`` und ``port`` angeben, so dass diese über das Netzwerk abgefragt werden können.


Aufbereitungsserver
-------------------

Das Spoolmodul enthält einen *Monitor*, der das Dateisystem überwacht, die Änderungen in Ihren Projekten und Konfigurationen überprüft und ggf. einen Hot-Reload des Servers durchführt. Sie können Intervalle für diese Prüfungen konfigurieren. Es wird empfohlen, das Monitorintervall auf mindestens 30 Sekunden einzustellen, da Dateisystemprüfungen ressourcenintensiv sind.
