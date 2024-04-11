# D-ProCon Integration :/admin-de/config/dprocon

TODO! %reference_de 'gws.ext.action.dprocon.Config'

Die GBD WebSuite kann mit dem System "D-ProCon" der Firma Sisterhenn (https://www.sisterhenn-it.de) integriert werden. Dafür muss eine Aktion vom Typ ``dprocon`` im Abschnitt ``actions`` eingetragen werden. Es gibt folgende Konfigurationsoptionen:

| OPTION | BEDEUTUNG |
|---|---|
| ``cacheTime`` | Cache Laufzeit für Ergebnisse |
| ``dataTable`` | Datenbank Tabelle für Ergebnisse |
| ``dataTablePattern`` | Pattern für die von D-ProCon angelegten Ergebnis-Tabellen |
| ``infoTitle`` | Title für das Informationsfenster |
| ``requestTableName`` | Name der Datenbank Tabelle für Anfragen |
| ``requestUrl`` | D-ProCon Aufruf Url |


## GBD WebSuite -> D-ProCon Anbindung

Der Nutzer wählt auf der Karte einen Bereich aus, und klickt den D-ProCon Button. Die Auswahl wird an den GBD WebSuite Server gesendet. Der Server führt eine räumliche Anfrage aus und ermittelt die Punkte (Häuser) die sich in der Auswahl befinden. Die Punkte werden in die Postgis Tabelle ``request`` geschrieben, die folgende Struktur hat:

```sql

    CREATE TABLE request (
        id SERIAL PRIMARY KEY,
        request_id CHARACTER VARYING,
        meso_key CHARACTER VARYING,
        ...andere Spalten für Referenzzwecken...
        selection geometry(GEOMETRY, 25832) NOT NULL,
        ts TIMESTAMP WITH TIME ZONE
    )
```

Danach wird D-ProCon in einem neuen Fenster geöffnet, mit der Angabe der ID.

## D-ProCon -> GBD WebSuite Anbindung

D-ProCon Ergebnisse werden in DB Tabellen bzw Views geschrieben. Diese Tabellen können ein beliebige Struktur haben, die Anforderung des GBD WebSuite ist, dass diese Tabellen eine Spalte ``request_id`` enthalten, die der Abfrage-ID entspricht und dass der Name der Tabelle dem ``dataTablePattern`` entspricht.

Außerdem muss im System ein QGIS-Projekt vorhanden sein ("Demografie-Projekt"), wo diese Tabellen/Views eingebunden sind. Hier ist die Anforderung der GBD WebSuite, dass diese Datenquellen mit dem Objektfilter ``request_id IS NOT NULL`` versehen werden.

Die "D-ProCon -> GBD WebSuite" Anbindung wird mit dieser URL aufgerufen

    http://example.com/project/demografie_project?demografie=<request_id>

Der Server macht eine Kopie des Demografie-Projekts und ersetzt dynamisch alle ``NOT NULL`` Filter mit dem Filter ``request_id = <angegebene request_id>``.

Danach wird die Kopie des Projekts im Browser aufgerufen, sodass der Nutzer nur die Daten zu sehen bekommt die der angegebenen Request-ID entsprechen. Die Layer bzw Datenquellen im Demografie-Projekt die keinen Filter haben (z.B. Hintergrund), werden unverändert dargestellt.

## Datenvorbereitung

TODO! ^CLIREF dprocon.setup

Bevor die D-ProCon Anbindung genutzt werden kann, sowie nach jeder ALKIS Aktualisierung, müssen die Daten mit dem  Befehl ``gws dprocon setup`` vorbereitet werden.
