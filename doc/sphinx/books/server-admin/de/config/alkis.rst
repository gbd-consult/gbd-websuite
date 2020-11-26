ALKIS Integration
=================

Die GBD WebSuite kann die Daten aus dem Amtliches Liegenschaftskatasterinformationssystem  (ALKIS) durchsuchen und bearbeiten. Es steht eine Client-Oberfläche für Flurstücksuche zur Verfügung und ein QGIS Plugin für ALKIS-basierte Geocodierung.

Um die ALKIS Integration zu nutzen benötigen Sie folgendes:

- eine Postgres/PostGIS Datenbank, die aus ALKIS Quelldaten (*NAS-Format*) erstellt wurde, z.B. mittels ALKIS-Import Software der Firma Norbit (http://www.norbit.de/68/)
- eine Helper Konfiguration
- Aktion Konfigurationen für die Flurstücksuche (``alkissearch``) und/oder Geocodierung (``alkisgeocoder``)

Helper ``alkis``
----------------

^REF gws.ext.helper.alkis.Config

In diesem Helper (s. ^helper) werden allgemeine ALKIS Einstellungen konfiguriert:

{TABLE}
   ``crs`` | KBS für ALKIS Daten, normalerweise ``EPSG:25832`` oder ``EPSG:25833``
   ``dataSchema`` | Postgres Schema in dem die ALKIS Tabellen liegen
   ``excludeGemarkung`` | Liste von Gemarkungen IDs (*gemarkungsnummer*), die aus der Suche ausgeschlossen werden müssen
   ``indexSchema`` | Postgres Schema in dem die Indexe geschrieben werden
{/TABLE}

Indizierung
-----------

^CLIREF alkis.create-index
^CLIREF alkis.drop-index

Bevor die ALKIS Daten für die Flurstücksuche verwendet werden können, müssen sie für GWS speziell indiziert werden. Dies erfolgt mit folgenden Kommandozeilen Befehlen:

- ``gws alkis drop-index`` - GWS Indizien löschen
- ``gws alkis create-index`` - GWS Indizien erzeugen

Die Index Tabellen werden in das unter ``indexSchema`` angegebene Schema geschrieben. Das ALKIS Modul schreibt nie in das ALKIS Schema (``dataSchema``). Diese Befehle müssen nach jeder ALKIS-Aktualisierung erneut ausgeführt werden. Aus Sicherheitsgründen werden bei diesen Befehlen Datenbank Benutzer- Kennung und Passwort explizit abgefragt.

Aktion ``alkissearch``
----------------------

^REF gws.ext.action.alkissearch.Config

Die Optionen für diese Aktion sind:

{TABLE}
    ``buchung`` | Zugang zu Grundbuchung Daten (z.B. Blattnummer)
    ``eigentuemer`` | Zugang zu Eigentümerdaten (z.B. Name, Adresse)
    ``export`` | Konfiguration der CSV-Export Funktion. Für diese Funktion muss auch den CSV Helper konfiguriert werden (s. ^csv)
    ``featureTemplates`` | Format-Vorlagen für Flurstückdaten
    ``limit`` | max. Anzahl von Ergebnissen
    ``printTemplate`` | Druckvorlage für Flurstückdaten
    ``ui`` | Einstellungen der Benutzeroberfläche
{/TABLE}

Einstellungen der Benutzeroberfläche
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Einstellungen der Benutzeroberfläche sind wie folgt:

{TABLE}
    ``autoSpatialSearch`` | räumliche Suche nach dem Absenden des Formulars aktivieren
    ``gemarkungListMode`` | Darstellung der Gemarkungsliste: ``plain`` = nur Gemarkungen, ``combined`` = "Gemarkung (Gemeinde)", ``tree`` = Baumansicht mit Gemeinden und Gemarkungen
    ``searchSelection`` | Funktion "In der Auswahl suchen" aktivieren
    ``searchSpatial`` | räumliche Suche freischalten
    ``strasseListMode`` | Darstellung der Straßen-Liste: ``plain`` = nur Straßennamen zeigen, ``withGemeinde`` oder ``withGemarkung`` = Straßenname + Gemeinde (oder Gemarkung),  ``withGemeindeIfRepeated`` / ``withGemarkungIfRepeated`` =  Straßenname + Gemeinde (Gemarkung) nur wenn der Name mehrmals vorkommt
    ``strasseSearchMode`` | Suchverhalten der Straßen-Liste (``start`` = nur Anfangssuche, ``all`` = überall suchen)
    ``useExport`` | CSV-Export Funktion freischalten
    ``usePick`` | Funktion "Flurstück direkt auswahlen" freischalten
    ``useSelect`` | Funktion "Flurstück selektieren" und die Ablage freischalten
{/TABLE}

Außerdem muss im Client-Einstellungen (s. ^client) das Element ``Sidebar.Alkis`` aktiviert werden.

Vorlagen
~~~~~~~~

Sie können Vorlagen mit Subjekten ``feature.title``, ``feature.teaser`` (Listenansicht) und ``feature.description`` (Detailsansicht) bei Bedarf anpassen. Die Standardvorlagen finden Sie unter https://github.com/gbd-consult/gbd-websuite/tree/master/app/gws/ext/action/alkissearch/templates zu finden. Ebenso kann mit ``printTemplate`` eine Druckvorlage angepasst werden.

Zugang zu Eigentümerdaten
~~~~~~~~~~~~~~~~~~~~~~~~~

Es besteht die Möglichkeit, den Zugang zu Eigentümerdaten für bestimmte Nutzerrollen einzugrenzen. Zusätzlich kann das Kontrolmodus (``controlMode``) aktiviert werden, wobei alle Zugriffe auf Eigentümerdaten auf Plausibilität geprüft und protokolliert werden. Die Plausibilitätsprüfung erfolgt indem das Formularfeld "Abrufgrund" mit angegebenen Regulären Ausdrucken verglichen wird. Eine Beispielkonfiguration kann wie folgt aussehen ::

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

Die Protokoll-Tabelle muss im System vorhanden sein, mit der folgender Struktur ::

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
        fs_ids TEXT
    )

Der Datenbank-Nutzer muss ``INSERT`` Recht auf diese Tabelle haben, aber nicht unbedingt ``SELECT``.

Aktion ``alkisgeocoder``
------------------------

^REF gws.ext.action.alkisgeocoder.Config

Für diese Aktion sind keine spezielle Optionen vorhanden. Da diese Aktion über unser QGIS-Plugin aufgerufen wird und über keine UI verfügt, müssen Sie die Autorisierungsmethode ``basic`` im System freischalten wenn Sie diese Aktion mit einem Passwort schützen möchten. Siehe dazu ^auth.
