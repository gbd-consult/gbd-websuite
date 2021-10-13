.. _inspire:

INSPIRE
=======

Das INSPIRE Modul der GBD WebSuite ermöglicht die dynamische Harmonisierung von Geodaten und deren Metadaten, in eine INSPIRE-konforme Datenbereitstellung. Damit die Eingabe der Daten in einem GIS einheitlich geschieht, können Attributformulare definiert und vorgegeben werden. Dadurch ist die Eingabe der Daten in einem einheitlichen Format garantiert. Dank dieser einheitlichen Struktur kann die WebSuite, eine Harmonisierung in eine INSPIRE-konforme Datenbereitstellung durchführen. Diese Harmonisierung auf INSPIRE-Konformität findet auf dem GBD WebSuite Server statt. Die Bereitstellung der Geodaten erfolgt auf Anfrage dynamisch durch den Server auf Basis von XML−Schemas. Das bedeutet, dass keine temporären, redundanten Daten erstellt und auf dem Server abgelegt werden. Die INSPIRE-konformen Dienste basieren immer auf den aktuellen Originaldaten. Für Transformation von Gauß−Krüger nach ETRS89 ist ein geeigneter Transformationsansatz integriert. Das Modul ist dabei so konzipiert, dass es zukünftig um weitere INSPIRE-Fachthemen erweitert werden kann. Wenn mehrere Fachthemen vorhanden sind, kann im Vorfeld das Fachthema ausgewählt und den Daten zugewiesen werden.

Diese Datenharmonisierung stellt das Grundmodul dar und kann generisch auf eine Vielzahl von Themen übertragen werden. Wir haben diese Funktionalität bisher für zwei INSPIRE-Themen umgesetzt. Für den Landkreis Marburg-Biedenkopf haben wir eine INSPIRE-konforme Datenharmonisierung, für die Themen Bauleitpläne und Schulstandorte umgesetzt. 

Unterstützt wird die Bereitstellung folgender INSPIRE Dienste:

- INSPIRE konformer Catalogue Service for the Web (CSW)
- INSPIRE konformer Web Mapping Service (WMS/WMTS)
- Umsetzung INSPIRE konformer Web Feature Service (WFS)
- Umsetzung INSPIRE konformer Web Coverage Service (WCS)



 .. |bplan| image:: ../../../images/bplan.svg
   :width: 30em

