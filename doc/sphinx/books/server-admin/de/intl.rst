Internationalisierung und Lokalisierung
==========================================

Der GBD WebSuite Server und Client sind sprach- und ortsunabhängig. Alle Spracheinstellungen sind frei konfigurierbar. In der Hauptkonfiguration haben wir die Standardoptionen ``locale`` und ``timeZone``. Zusätzlich können Sie das Gebietsschema für jedes Projekt individuell einstellen.

Alle Anfragen an den GBD WebSuite Server und alle Serverantworten werden als ``UTF-8`` verschlüsselt. Andere Kodierungen werden von uns nicht unterstützt.

Beispiel für die Gebietsschema-Konfiguration ::

    ## in der Hauptkonfiguration:

    "locale": "de_DE",
    "timeZone": "Europe/Berlin"

Für ``mako`` Templates bieten wir lokale ``date`` und ``time`` Objekte, mit den Methoden ``long``, ``medium`` und ``short``. Es folgen Ausgabebeispiele für das Gebietsschema ``de_DE``:

TABLE
    ``date.short()`` ~ 08.12.18
    ``date.medium()`` ~ 08.12.2018
    ``date.long()`` ~ 8\. Dezember 2018
    ``time.short()`` ~ 19:35
    ``time.medium()`` ~ 19:35:59
    ``time.long()`` ~ 19:35:59 +0000
/TABLE
