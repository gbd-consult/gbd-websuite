.. _search:

Suche
=====

**Funktion:**

Mit der |search| :guilabel:`Suche` Funktion ist es möglich, in Attributwerten von Objekten zu suchen.
Diese Objekte werden dann aufgelistet und können so überblickt werden.

**Bedienung:**

Durch die verschiedenen Konfigurationsmöglichkeiten der GBD WebSuite kann die |search| :guilabel:`Suche` in die Menüleiste,
sowie in die Werkzeugleiste als Menüpunkt :ref:`Suchen <searching>` integriert sein.
Die Möglichkeiten und Optionen dazu sind im "GBD WebSuite Server Administrator Handbuch" beschrieben.
Je nachdem ob Sie die Suche über die Menü- oder Werkzeugleiste aktivieren, öffnet sich auch eine Suchleiste an dazugehöriger Stelle.

In diese können Sie klicken und Ihren gesuchten Begriff eingeben.
Die Ergebnisse werden unterhalb der Suchleiste aufgelistet.
Durch ein Anklicken eines Ergebnis, wird das dazugehörige Objekt automatisch im Kartenfenster fokussiert
und es öffnen sich die dazugehörigen Objekteigenschaften in einem Pop-up Fenster.

.. figure:: ../../../screenshots/de/client-user/search_menue.png
  :align: center

* Die Suche wird unterstützt durch die Eingabe von Anfangsbuchstaben, freien Texten und Zahlenwerten.
* Die integrierte Autocomplete-Funktion sorgt für eine dynamische Suche mit simultaner Anzeige der Ergebnisse.
* Die Suche kann Nominatim, sprich OSM-Daten, sowie Postgres-Datenbanken durchsuchen.

  * Bei der Nominatim-Suche werden die Objekteigenschaften von OSM-Objekten genutzt.
  * Bei der Suche in Postgres-Datenbanken wird auf Informationen einzelner Spalten der Datenbank zugegriffen.

.. admonition:: Konfigurationsmöglichkeiten

 Durch Anpassungen der Konfiguration ist eine Beschränkung der Suche auf bestimmte Attribute möglich.

 .. |search| image:: ../../../images/baseline-search-24px.svg
    :width: 30em
