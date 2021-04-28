Internationalisierung und Lokalisierung
=======================================

Die GBD WebSuite Server und Client sind sprach- und ortsunabhängig, alle Spracheinstellungen sind konfigurierbar. In der Konfiguration haben wir die Standardoption ``locales``, zusätzlich können Sie das Gebietsschema für jedes Projekt individuell einstellen.

Alle Anfragen an den GBD WebSuite Server und alle Serverantworten werden als ``UTF-8`` kodiert. Andere Kodierungen werden von uns nicht unterstützt.

Beispiel für die Gebietsschema-Konfiguration ::

    "locales": ["de_DE", "en_US"]

Für Vorlagen bieten wir lokale ``date`` und ``time`` Objekte, mit den Eigenschaften ``long``, ``medium`` und ``short``. Ausgabebeispiele für das Gebietsschema ``de_DE``:

{TABLE}
    ``date.short`` | 08.12.18
    ``date.medium`` | 08.12.2018
    ``date.long`` | 8\. Dezember 2018
    ``time.short`` | 19:35
    ``time.medium`` | 19:35:59
    ``time.long`` | 19:35:59 +0000
{/TABLE}
