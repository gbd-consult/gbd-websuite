D-ProCon Integration
====================

Die Funktionalität von "D-ProCon" von der Firma Sisterhenn (https://www.sisterhenn-it.de) kann in die GBD WebSuite integriert werden. Dafür muss eine Aktion vom Typ ``dprocon`` im Abschnitt ``actions`` eingetragen werden. Es gibt folgende Konfigurationsoptionen:

TABLE
*alkisSchema* ~ ALKIS Datenschema
*cacheTime* ~ Cache Laufzeit für Ergebnisse
*crs* ~ KBS für ALKIS Daten
*dataTable* ~ Datenbank Tabelle für Ergebnisse
*dataTablePattern* ~ Pattern für die von D-ProCon angelegten Ergebnis-Tabellen
*db* ~ Datenbank Anbieter
*indexSchema* ~ ALKIS Indexschema
*infoTitle* ~ Titel für das Informationsfenster
*requestTable* ~ Datenbank Tabelle für Anfragen
*requestUrl* ~ D-ProCon Aufruf Url
/TABLE

GBD WebSuite -> D-ProCon Anbindung
----------------------------------

Der Nutzer wählt auf der Karte einen Bereich aus und klickt auf den D-ProCon Button.
Die Auswahl wird an den GBD WebSuite Server gesendet. Der Server führt eine räumliche Anfrage aus und ermittelt die Punkte (Häuser) die sich in der Auswahl befinden. Die Punkte werden in die Postgis Tabelle ``request`` geschrieben, im folgenden Format:

TABLE
``request_id`` ~ eindeutige Anfrage ID
``meso_key`` ~ MESO key
... ~ andere Spalten für Referenzzwecke
/TABLE

Danach wird D-ProCon,, mit der Angabe der ID, in einem neuen Fenster geöffnet.


D-ProCon -> GBD WebSuite Anbindung
----------------------------------

Die D-ProCon Ergebnisse werden in Datenbanktabellen bzw Views geschrieben. Diese Tabellen können eine beliebige Struktur haben. Die Anforderung der GBD WebSuite ist lediglich, dass diese Tabellen eine Spalte ``request_id`` enthalten, die der Abfrage-ID entspricht und dass der Name der Tabelle dem ``dataTablePattern`` entspricht.

Außerdem muss im System ein QGIS-Projekt vorhanden sein ("Demografie-Projekt"), wo diese Tabellen/Views eingebunden sind.
Hier ist die Anforderung der GBD WebSuite, dass diese Datenquellen mit dem Objektfilter ``request_id IS NOT NULL`` versehen werden.

Die "D-ProCon -> GBD WebSuite" Anbindung wird mit dieser URL aufgerufen ::

    http://server-name/project/demografie_project?demografie=<request_id>

Der Server macht eine Kopie des Demografie-Projekts und ersetzt dynamisch alle ``NOT NULL`` Filter mit dem Filter ``request_id = <agegebene request_id>``.

Danach wird die Kopie des Projekts im Browser aufgerufen, sodass der Nutzer nur die Daten zu sehen bekommt, die der angegebenen Request-ID entsprechen. Die Layer bzw Datenquellen im Demografie-Projekt, die keinen Filter haben (z.B. Hintergrund), werden unverändert dargestellt.

Datenvorbereitung
-----------------

Bevor die D-ProCon Anbindung genutzt werden kann, sowie nach jeder ALKIS Aktualisierung, müssen die Daten mit diesem Befehl vorbereitet werden ::

    gws dprocon setup
