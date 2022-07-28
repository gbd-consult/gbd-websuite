Suche
=====

Mit der |search| :guilabel:`Suche` Funktion ist es möglich, in Attributwerten von Objekten zu suchen.
Diese Objekte werden dann aufgelistet und können so überblickt werden.

Wenn Sie den Menüpunkt auswählen, öffnet sich eine Suchleiste.
In diese können Sie klicken und Ihren gesuchten Begriff eingeben.
Die Ergebnisse werden unterhalb der Suchleiste aufgelistet.
Durch ein Anklicken eines Ergebnis, wird das dazugehörige Objekt automatisch im Kartenfenster fokussiert
und es öffnen sich die dazugehörigen Objekteigenschaften in einen Pop-up Fenster.

.. figure:: ../../../screenshots/de/client-user/search_menue.png
  :align: center

.. tip::
 Unterstützt wird die Suche durch Eingabe von Anfangsbuchstaben, freien Texten und Zahlenwerten.
 Die integrierte Autocomplete-Funktion sorgt für eine dynamische Suche sowie dafür, dass Ihnen die Ergebnisse direkt angezeigt werden.
 Die Suche kann Nominatim, sprich OSM-Daten, sowie Postgres-Datenbanken durchsuchen.
 Bei der Nominatim-Suche werden die Objekteigenschaften von OSM-Objekten genutzt.
 Bei der Suche in Postgres-Datenbanken, wird auf Informationen einzelner Spalten der Datenbank zugegriffen.

.. note:: **Konfigurationsmöglichkeiten**
 Die verwendeten Suchparameter können definiert werden.

 .. |search| image:: ../../../images/baseline-search-24px.svg
   :width: 30em
