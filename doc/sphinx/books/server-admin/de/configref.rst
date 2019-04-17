Konfigurationsreferenz
=======================

Dieser Abschnitt beschreibt die GBD WebSuite Konfiguration, bei der es sich um eine geschachtelte Key-Value-Struktur handelt. Die *keys* sind Zeichenketten, die *values* sollten zu einem der unten aufgeführten Typen gehören.

Die Top-Level-Konfiguration ist vom Typ **ApplicationConfig**.

Basis Typen
-----------

Grundlegende Datentypen, wie sie in Python verwendet werden:

TABLE
   ``str`` ~ String, muss in der UTF-8 Kodierung sein
   ``bool`` ~ Boolean true or false
   ``int`` ~ Integer Nummer
   ``float`` ~ Fließkommazahl
   ``dict`` ~ Generisches Schlüssel-Wert-Objekt
   [``T``...] ~ Liste (Array) von Elementen vom Typ  ``T``
/TABLE

Spezial Typen
-------------

Die Werte dieser Typen sind Zeichenketten und Zahlen, die mit einer speziellen Semantik versehen sind.


.. include:: /{DOC_ROOT}/gen/de.configref.txt
