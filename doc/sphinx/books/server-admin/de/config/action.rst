Server Aktionen
===============

%reference_de 'gws.types.ext.action.Config'

Die GBD WebSuite unterstützt viele verschiedene Befehle. In Ihrer App- und Projektkonfiguration können Sie entscheiden, welche Sie in Ihrer speziellen Installation verwenden möchten.
Hier ein kurzer Überblick über die unterstützten Aktionstypen:

{TABLE}
    ``alkisgeocoder`` | Schnittstelle zur Geocodierung auf dem Basis der deutschen Katasterdaten (s. ^alkis)
    ``alkissearch`` | Suche nach Katasterparzellen (*Flurstücke*) nach Adresse, Standort, Besitzername etc. (s. ^alkis)
    ``asset`` | Verwaltung von dynamischen Assets (s. ^web)
    ``auth`` | Autorisierung-Anfragen, wie Login oder Logout (s. ^auth)
    ``bplan`` | Verwaltung von Bauleitplänen (s. ^bplan)
    ``dimension`` |  Bemaßung-Funktion (s. ^dimension)
    ``dprocon`` | DPro-Con Anbindung(s. ^dprocon)
    ``edit`` | Digitalisierung (s. ^digitize)
    ``fs`` | Schnittstelle für das virtuelle Dateisystem, mit der die Daten auf dem Server gespeichert werden können
    ``gekos`` | GekoS Anbindung (s. ^gekos)
    ``map`` | Render von Kartenbildern (s. ^map)
    ``ows`` | OWS Dienste wie WMS, WMTS oder WFS (s. ^ows)
    ``print`` | Drucken (s. ^print)
    ``project`` | Projektbeschreibung und Konfiguration für den Client (s. ^project)
    ``search`` | allgemeine Such-Anfragen (s. ^search)
    ``storage`` | Datenaustausch mit der Datenablage (s. ^storage)
    ``tabedit`` | Editieren von Sachdaten in einer tabellarischen Form (s. ^tabedit)
{/TABLE}

Aktionen werden in ``api`` Blocks in der App- bzw Projektkonfigs konfiguriert. Jede Aktionskonfiguration benötigt mindestens die Eigenschaft ``type``, und optional einen ``access`` Block (s. ^auth), der angibt, welche Rollen die Berechtigung haben, diesen Aktionstyp auszuführen. Einige Aktionen erfordern zusätzliche Optionen.
