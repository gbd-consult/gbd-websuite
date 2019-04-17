ALKIS Integration
=================

Die GBD WebSuite kann auf Daten aus dem amtlichen Liegenschaftskatasterinformationssystem (ALKIS) zugreifen und diese durchsuchen und bearbeiten. Unter anderem steht ein Plugin zur Flurstückssuche zur Verfügung. Die Konfiguration erfolgt im Abschnitt ``actions``, Typ ``alkis``. Folgende Optionen sind hier verfügbar:

TABLE
    *alkisSchema* ~ PostGIS Schema wo die ALKIS Daten hinterlegt sind
    *buchung* ~ Zugang zu Grundbuch Daten (z.B. Blattnummer)
    *db* ~ Datenbank-Provider ID
    *eigentuemer* ~ Zugang zu Eigentümerdaten (z.B. Name, Adresse)
    *excludeGemarkung* ~ Liste von Gemarkungen, die aus der Suche ausgeschlossen werden müssen
    *featureFormat* ~ Formatierung von Flurstücksdaten
    *indexSchema* ~ PostGIS Schema wo die Indexe geschrieben werden
    *limit* ~ Anzahl von Ergebnissen
    *printTemplate* ~ Druckvorlage für Flurstücksdaten
    *ui* ~ Einstellungen der Benutzeroberfläche
/TABLE

Einstellungen der Benutzeroberfläche sind wie folgt:

TABLE
    *export* ~ CSV-Export Funktion aktivieren
    *pick* ~ Funktion "Flurstück direkt auswahlen" aktivieren
    *searchSelection* ~ Funktion "In der Auswahl suchen" aktivieren
    *searchSpatial* ~ Räumliche Suche aktivieren
    *select* ~ Auswahl-Funktion aktivieren
/TABLE


Indizierung
-----------

Bevor die ALKIS Daten für die Flurstückssuche verwendet werden können, müssen sie für die GBD WebSuite speziell indiziert werden. Dies ist mit folgenden Kommandozeilen Befehlen möglich ::


    ## gws Indizien löschen

    gws alkis drop-index


    ## gws Indizien erzeugen

    gws alkis create-index


Diese Befehle müssen nach jeder ALKIS-Aktualisierung erneut ausgeführt werden.


Vorlagen
--------

Es sind folgende Standardvorlagen im Plugin vorhanden:

TABLE
teaser.cx.html ~ Vorlage für die Beschriftung in der Ergebnissliste
data.cx.html ~ Vorlage für die Flurstückdetails
print.cx.html ~ Druckvorlage
/TABLE

Diese Vorlagen sind unter https://github.com/gbd-consult/gbd-websuite/tree/master/app/gws/ext/action/alkis/templates zu finden. Für die Anpassung einer Vorlage ist es zu empfehlen eine Kopie der Standardvorlage anzulegen und die ``featureFormat`` Option entsprechend anzupassen, z.B. ::


    "featureFormat": {
        "description": {
            "type": "html",
            "path": "/data/vorlagen/meine-fs-details-vorlage.cx.html"
        }
    }


Zugang zu Eigentümerdaten
-------------------------

Es besteht die Möglichkeit, den Zugang zu Eigentümerdaten für bestimmte Nutzerrollen einzugrenzen. Zusätzlich kann der Kontrolmodus (``controlMode``) aktiviert werden, wobei alle Zugriffe, auf Eigentümerdaten, auf Plausibilität geprüft und protokolliert werden. Eine Beispielkonfiguration kann wie folgt aussehen ::

    "eigentuemer": {

        ## Zugang nur für "vorstand" zulassen

        "access": [
            {"type": "allow", "role": "vorstand"},
            {"type": "deny", "role": "all"}
        ],

        ## Kontrolmodus aktivieren

        "controlMode": true,

        ## Regel für Plausibilitätsprüfung

        "controlRules": [
            ## ein Aktenzeichen im Format 2 Buchstaben / 2 Zahlen
            "^[A-Z][A-Z]/[0-9][0-9]$"
        ],

        ## Postgis Tabelle für Protokollierung

        "logTable": "eigen_log"
    }

Die Protokoll-Tabelle muss mit der folgenden Struktur, im System vorhanden sein ::

    CREATE TABLE .... (
        id SERIAL PRIMARY KEY,
        app_name VARCHAR(255),
        date_time TIMESTAMP,
        ip VARCHAR(255),
        login VARCHAR(255),
        user_name VARCHAR(255),
        control_input VARCHAR(255),
        control_result INTEGER,
        fs_count INTEGER,
        fs_ids VARCHAR(255)
    )
