.. _print:

Drucken
=======

Mithilfe des |print| ``Drucken``-Werkzeugs kann eine PDF generiert werden, die gespeichert und gedruckt werden kann.
Die GBD WebSuite kann in QGIS erstellte Druckvorlagen übernehmen und in Form einer Auswahl bereitstellen.
Beim Starten der Druckfunktion wird der aktuelle Kartenausschnitt unverändert übernommen.
Auf der Karte gezeichnete, beschriftete oder durch eine Abfrage markierte Objekte werden dabei in die Druckkarte übernommen.
Globale Einstellung wie die festgelegte :ref:`Rotation <turn>` und maßstabsbezogene Ebenendarstellungen werden beim Druck mit übernommen.
Auch manuell festgelegte Transparenzen für einzelne Layer und Layergruppen werden übernommen.

Beim Aktivieren der Druckfunktion öffnet sich auf der Karte ein Kartenrahmen, der den zu druckenden Auswahlbereich symbolisiert.
Dieser kann beliebig verschoben werden.
Mit der linken Maustaste und dem Mausrad kann der entsprechende Ausschnitt und die Zoomstufe eingestellt werden.
Bei Bedarf kann auch ein Titel manuell eingefügt werden und die Druckqualität bestimmt werden. Wählen Sie dazu einen Wert zwischen 70 und 300 dpi.

 .. figure:: ../../../screenshots/de/client-user/print1.png
   :align: center

Neben dem Kartenrahmen öffnet sich ebenfalls ein Dialogfenster, in dem die Ausrichtung der Druckvorlage sowie die Druckauflösung eingestellt werden können.
Wird das Drucksymbol |print| erneut gedrückt, öffnet sich die druckfertige PDF. Mit dem QGIS Drucklayout erstellte Legende können hier bereitgestellt und eingebunden werden. Generell können nun PNG und HTML Legenden verwendet werden um im Druck eingebunden zu werden.
So kann man bei Layern der GBD WebSuite welche keine eigenen Legenden besitzen, Legenden zur druckbaren Karte hinzufügen.

 .. |print| image:: ../../../images/baseline-print-24px.svg
   :width: 30em
 .. |cancel| image:: ../../../images/baseline-cancel-24px.svg
   :width: 30em

.. .. figure:: ../../../screenshots/de/client-user/print_2.png
      :scale: 60%
      :align: center