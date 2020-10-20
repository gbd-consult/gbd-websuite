Konfigurationsreferenz
======================

Dieser Abschnitt beschreibt formal die GBD WebSuite Konfiguration, bei der es sich um eine geschachtelte Schlüssel-Wert-Struktur handelt. Die *Schlüssel* sind immer Zeichenketten, die *Werte* sollten zu einem der unten aufgeführten Typen gehören.

Der Typ :ref:`de_configref_gws_common_application_Config` präsentiert die Hauptkonfiguration.

Basis Typen
-----------

Grundlegende Datentypen sing wie folgt:

{TABLE}
   ``str`` | String, muss in der UTF-8 Kodierung sein
   ``bool`` | Boolean true or false
   ``int`` | integer Nummer
   ``float`` | reelle Nummer
   ``dict`` | generisches Schlüssel-Wert-Objekt
   **[** ``Typ`` **]** | Liste (Array) von Elementen vom ``Typ``
{/TABLE}

.. include:: ../../../../ref/de.configref.txt
