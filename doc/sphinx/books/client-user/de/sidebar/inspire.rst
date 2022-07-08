.. _inspire:

INSPIRE
=======

INSPIRE (Infrastructure for Spatial Information in Europe) steht als Kürzel für eine EU-Richtlinie zur Schaffung einer Geodateninfrastruktur, auf die über standardisierte Dienste interoperabel zugegriffen werden kann. Es gibt 34 INSPIRE Themenfelder, die grundlegende Basisinformationen über Geographische Namen, Adressen oder Grundstücke sowie fachspezifische Informationen z.B. zu den Themen Gesundheit, Bevölkerung oder Bodenschätze bereitstellen.

.. rubric:: INSPIRE-konforme Daten- und Metadatenbereitstellung

Das INSPIRE Modul der GBD WebSuite ermöglicht eine dynamische, INSPIRE-konforme Bereitstellung von INSPIRE Themen als Geodaten und deren Metadaten. Dynamisch bedeutet dabei, dass die Bereitstellung des angeforderten INSPIRE-Dienstes bei Anfrage, also "in Echtzeit", aus dem originalem Datenbestand stattfindet. Damit wird sichergestellt, dass die INSPIRE-konformen Dienste immer dem aktuellen Stand entsprechen und dass keine redundante Datenhaltung notwendig ist.

Unterstützt wird aktuell die Bereitstellung folgender INSPIRE Dienste:

- INSPIRE-konformer Web Mapping Service (WMS/WMTS)
- INSPIRE-konformer Web Feature Service (WFS)
- INSPIRE-konformer Web Coverage Service (WCS)
- INSPIRE-konformer Catalogue Service for the Web (CSW)

Die INSPIRE-Datenharmonisierung, d.h die Überführung bestehender, originaler Geodatenbestände in die erforderlichen INSPIRE-Datenstrukturen und Metadaten, findet im INSPIRE Modul der GBD WebSuite automatisch mittels einer individuell konfigurierbaren Datenharmonisierung statt. Die Aktualisierung der Geodaten und der dazugehörigen Metadaten kann bei Bedarf aber auch über ein Administrationstool ermöglicht werden.

Für die Datentransformation, z.B. von Gauß−Krüger nach ETRS89, sind geeignete Transformationsansätze integriert.

.. rubric:: INSPIRE Themen

Das INSPIRE Modul der GBD WebSuite ist generisch umgesetzt und so konzipiert, dass es um weitere INSPIRE-Fachthemen erweitert werden kann.

Unterstützt werden aktuell folgende INSPIRE-Themen:

- Bauleitpläne (`Beispielumsetzung beim LK Marburg-Biedenkopf <https://www.marburg-biedenkopf.de/dienste_und_leistungen/geoportal/Bauleitplaene-Landkreis-Marburg-Biedenkopf.php>`_)
- Schulstandorte (`Beispielintegration Schulstandorte des LK Marburg-Biedenkopf in das Geoportal Hessen <https://www.geoportal.hessen.de/map?WMC=4768>`_)
- Feuerwehrstandorte und Einsatzbereiche (bis Ende 2022)
- Rettungswachen und Bezirke (bis Ende 2022)
- Kindertageseinrichtungen (bis Ende 2022)

 .. |bplan| image:: ../../../images/bplan.svg
   :width: 30em
