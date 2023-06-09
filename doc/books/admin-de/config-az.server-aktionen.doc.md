# Server Aktionen :/admin-de/config-az/server-aktionen

%reference_de 'gws.ext.config.action'

Die GBD WebSuite unterstützt viele verschiedene Befehle. In Ihrer App- und Projektkonfiguration können Sie entscheiden, welche Sie in Ihrer speziellen Installation verwenden möchten.
Hier ein kurzer Überblick über die unterstützten Aktionstypen:

| OPTION | BEDEUTUNG |
|---|---|
| ``alkisgeocoder`` | Schnittstelle zur Geocodierung auf dem Basis der deutschen Katasterdaten, siehe [Alkis](/admin-de/config-az/alkis) |
| ``alkissearch`` | Suche nach Katasterparzellen (*Flurstücke*) nach Adresse, Standort, Besitzername etc. siehe [Alkis](/admin-de/config-az/alkis) |
| ``asset`` | Verwaltung von dynamischen Assets, siehe [Web](/admin-de/config-az/web) |
| ``auth`` | Autorisierung-Anfragen, wie Login oder Logout, siehe [Autorisierung](/admin-de/config-az/autorisierung) |
| ``bplan`` | Verwaltung von Bauleitplänen, siehe [Bauleitplanung](/admin-de/config-az/bplan) |
| ``dimension`` |  Bemaßung-Funktion, siehe [Bemaßung](/admin-de/config-az/bemassung) |
| ``dprocon`` | DPro-Con Anbindung, siehe [D-ProCon](/admin-de/config-az/dprocon) |
| ``edit`` | Digitalisierung, siehe [Digitalisierung](/admin-de/plugin/edit) |
| ``fs`` | Schnittstelle für das virtuelle Dateisystem, mit der die Daten auf dem Server gespeichert werden können |
| ``gekos`` | GekoS Anbindung, siehe [GekoS Integration](/admin-de/config-az/gekos) |
| ``map`` | Render von Kartenbildern, siehe [Karten](/admin-de/config-az/karten) |
| ``ows`` | OWS Dienste wie WMS, WMTS oder WFS, siehe [OWS](/admin-de/config-az/ows) |
| ``print`` | Drucken, siehe [Drucken](/admin-de/config-az/drucken) |
| ``project`` | Projektbeschreibung und Konfiguration für den Client, siehe [Projekte](/admin-de/config-az/projekte) |
| ``search`` | allgemeine Such-Anfragen, siehe [Suche](/admin-de/config-az/suche) |
| ``storage`` | Datenaustausch mit der Datenablage, siehe [Suche](/admin-de/config-az/suche) |
| ``tabedit`` | Editieren von Sachdaten in einer tabellarischen Form, siehe [Tabellarisches Editieren](/admin-de/plugin/tabedit) |

Aktionen werden in ``api`` Blocks in der App- bzw Projektkonfigs konfiguriert. Jede Aktionskonfiguration benötigt mindestens die Eigenschaft ``type``, und optional einen ``access`` Block (siehe [Autorisierung](/admin-de/config-az/autorisierung)), der angibt, welche Rollen die Berechtigung haben, diesen Aktionstyp auszuführen. Einige Aktionen erfordern zusätzliche Optionen.
