Server Aktionen
===============

^REF gws.types.ext.action.Config

Die GBD WebSuite unterstützt viele verschiedene Befehle. In Ihrer Anwendungs- und Projektkonfiguration können Sie entscheiden, welche Sie in Ihrer speziellen Installation verwenden möchten. Jede Aktionskonfiguration benötigt mindestens die Eigenschaft ``type``, und optional einen ``access`` Block (s. ^auth), der angibt, welche Rollen die Berechtigung haben, diesen Aktionstyp *auszuführen*. Einige Aktionen erfordern zusätzliche Optionen.

Hier ein kurzer Überblick über die unterstützten Aktionstypen:

{TABLE}
    ``alkisgeocoder`` | bietet eine Schnittstelle zur Geocodierung auf dem Basis der deutschen Katasterdaten (s. ^alkis)
    ``alkissearch`` | Suche nach Katasterparzellen (*Flurstücke*) nach Adresse, Standort, Besitzername etc. (s. ^alkis)
    ``asset`` | verwaltet dynamische Assets (s. ^web)
    ``auth`` | bearbeitet Autorisierungsanfragen, wie Login oder Logout (s. ^auth)
    ``bplan`` | Verwaltung von Bauleitplänen (s. ^bplan)
    ``dimension`` |  (s. ^dimension)
    ``dprocon`` | (s. ^dprocon)
    ``edit`` | (s. ^digitize)
    ``fs`` | bietet eine Schnittstelle für das virtuelle Dateisystem, mit der die Daten auf dem Server gespeichert werden können
    ``gekos`` | (s. ^gekos)
    ``map`` | erzeugt Kartenbilder für Projekte und Ebenen in verschiedenen Formaten (s. ^map)
    ``ows`` | unterstützt OWS Dienste wie WMS, WMTS oder WFS (s. ^ows)
    ``printer`` | übernimmt das Drucken (s. ^print)
    ``project`` | gibt Projektbeschreibung und Konfigurationsdaten zurück (s. ^project)
    ``search`` | behandelt die Suche (s. ^search)
    ``storage`` | übernimmt den Datenaustausch mit der Datenablage (s. ^storage)
    ``tabedit`` | Editieren von Sachdaten in einer tabellarischen Form (s. ^tabedit)
{/TABLE}

Aktionen werden in ``api`` Blocks in der Applikation- bzw Projektkonfigs konfiguriert.
