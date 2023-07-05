# Server Aktionen :/admin-de/config/server-aktionen

%reference_de 'gws.ext.config.action'

Die GBD WebSuite unterstützt viele verschiedene Befehle. In Ihrer App- und Projektkonfiguration können Sie entscheiden, welche Sie in Ihrer speziellen Installation verwenden möchten.
Hier ein kurzer Überblick über die unterstützten Aktionstypen und deren Client Abhängigkeiten:

| OPTION | BEDEUTUNG | Client Seitenleiste | Client Werkzeugleiste |
|---|---|---|---|
| ``alkisgeocoder`` | Schnittstelle zur Geocodierung auf der Basis der deutschen Katasterdaten, siehe [Alkis](/admin-de/plugin/alkis) | | |
| ``alkissearch`` | Suche nach Katasterparzellen (*Flurstücke*) nach Adresse, Standort, Besitzername etc. siehe [Alkis](/admin-de/plugin/alkis) | [Suche](/user-de/sidebar.suche) | |
| ``annotate`` | vom Benutzer erstellte Markierungen | [Markierungen](/user-de/sidebar.markierung) | |
| ``auth`` | Autorisierung-Anfragen, wie Login oder Logout, siehe [Autorisierung](/admin-de/config/autorisierung) | [Anmelden](/user-de/sidebar.anmelden) | |
| ``dimension`` |  Bemaßung-Funktion, siehe [Bemaßung](/admin-de/plugin/dimension) | [Bemaßung](/user-de/sidebar.bemassung) | [Bemaßung](/user-de/toolbar.bemassung) |
| ``edit`` | Digitalisierung, siehe [Digitalisierung](/admin-de/plugin/edit) | [Digitalisieren](/user-de/sidebar.digitalisieren) | [Digitalisieren](/user-de/toolbar.digitalisieren) |
| ``map`` | Render von Kartenbildern, siehe [Karten](/admin-de/config/karten) | [Layer](/user-de/sidebar.layer) | |
| ``ows`` | OWS Dienste wie WMS, WMTS oder WFS, siehe [OWS](/admin-de/config/ows) | | |
| ``printer`` | Drucken, siehe [Drucken](/admin-de/config/drucken) | | [Drucken](/user-de/toolbar.drucken) |
| ``project`` | Projektbeschreibung und Konfiguration für den Client, siehe [Projekte](/admin-de/config/projekte) |[Porjektübersicht](/user-de/sidebar.projektuebersicht) | |
| ``search`` | allgemeine Such-Anfragen, siehe [Suche](/admin-de/config/suche) | [Suche](/user-de/sidebar.suche) | [Suche](/user-de/toolbar.suchen) & [Räumliche Suche](/user-de/toolbar.raeumliche-suche) |
| ``storage`` | Datenaustausch mit der Datenablage, siehe [Datenablage](/admin-de/config/datenablage) | [Markierungen](/user-de/sidebar.markierung) & [Bemaßung](/user-de/sidebar.bemassung) & [Auswahl](/user-de/sidebar.auswahl) | [Markieren](/user-de/toolbar.markieren-messen) & [Bemaßung](/user-de/toolbar.bemassung) & [Auswahl](/user-de/toolbar.auswahl) |
| ``web`` | liefert Webseiten aus, siehe [Web-Server](/admin-de/config/web) | | |


Aktionen werden in ``api`` Blocks in der App- bzw Projektkonfigs konfiguriert. Jede Aktionskonfiguration benötigt mindestens die Eigenschaft ``type``, und optional einen ``access`` Block (siehe [Autorisierung](/admin-de/config/autorisierung)), der angibt, welche Rollen die Berechtigung haben, diesen Aktionstyp auszuführen. Einige Aktionen erfordern zusätzliche Optionen.


**Zukünftige Aktionen**

| OPTION | BEDEUTUNG | Client Seitenleiste | Client Werkzeugleiste |
|---|---|---|---|
| ``asset`` | Verwaltung von dynamischen Assets, siehe [Web-Server](/admin-de/config/web) | | |
| ``bplan`` | Verwaltung von Bauleitplänen, siehe [Bauleitplanung](/admin-de/config/bplan) |[Bauleitplanung](/user-de/sidebar.bauleitplanung) | |
| ``dprocon`` | DPro-Con Anbindung, siehe [D-ProCon](/admin-de/config/dprocon) | |[D-ProCon](/user-de/toolbar.dprocon) |
| ``fs`` | Schnittstelle für das virtuelle Dateisystem, mit der die Daten auf dem Server gespeichert werden können | | |
| ``gekos`` | GekoS Anbindung, siehe [GekoS Integration](/admin-de/config/gekos) | | [GeKos](/user-de/toolbar.gekos) |
| ``tabedit`` | Editieren von Sachdaten in einer tabellarischen Form, siehe [Tabellarisches Editieren](/admin-de/plugin/tabedit) | [Tabellen editieren](/user-de/sidebar.tabellen) | |