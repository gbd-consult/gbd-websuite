Server Aktionen
================

Die GBD WebSuite unterstützt viele verschiedene Befehle. In Ihrer Anwendungs- und Projektkonfiguration können Sie entscheiden, welche Sie in Ihrer speziellen Installation verwenden möchten. Jede Aktionskonfiguration benötigt mindestens die Eigenschaft ``type`` (das erste Wort des Befehlsnamens, z. B. für den Befehl ``mapRenderXyz``, der Typ ist ``map``), und optional einen ``access`` Block (siehe :doc:`auth`), der angibt, welche Rollen die Berechtigung haben, diesen Aktionstyp *auszuführen*. Einige Aktionen erfordern zusätzliche Optionen.

Hier ein kurzer Überblick über die unterstützten Aktionen (siehe :doc:`configref` für Details zur Konfiguration).

TABLE
*alkis* ~ bietet eine Schnittstelle zu den deutschen Katasterdaten (`ALKIS <http://www.adv-online.de/Products/Real-Estate-Cadastre/ALKIS/>`_). Insbesondere gibt es Befehle zur Suche nach Katasterparzellen (*Flurstücke*) nach Adresse, Standort, Besitzername etc.
*asset* ~ verwaltet dynamische Assets (siehe :doc:`web`)
*auth* ~ bearbeitet Autorisierungsanfragen, wie Login oder Logout (siehe :doc:`auth`)
*edit* ~ stellt das Backend für die Bearbeitung von Operationen zur Verfügung (z. B. "Update-Funktion" oder "Löschfunktion")
*map* ~ erzeugt Kartenbilder für Projekte und Ebenen in verschiedenen Formaten
*printer* ~ übernimmt das Drucken
*projekt* ~ gibt Projektbeschreibung und Konfigurationsdaten zurück
*search* ~ behandelt die Suche (siehe :doc:`search`)
/TABLE

Wir pflegen die komplette Liste der Befehle, deren Argumente und Rückgabewerte als TypeScript-Schnittstellen-Datei.
