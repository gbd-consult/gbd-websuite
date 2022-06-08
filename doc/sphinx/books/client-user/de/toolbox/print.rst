.. _print:

Drucken
=======

Mithilfe des |print| ``Drucken``-Werkzeugs wird eine PDF generiert, die gespeichert und gedruckt werden kann.
Die GBD WebSuite kann in QGIS erstellte Druckvorlagen übernehmen und in Form einer Auswahl bereitstellen.
Außerdem können mit dem QGIS Drucklayout erstellte Legenden, bereitgestellt und eingebunden werden.
Es können als PNG oder HTML gespeicherte Legenden im Druck verwendet werden.
So kann man bei Layern der GBD WebSuite, welche keine eigenen Legenden besitzen, Legenden zur druckbaren Karte hinzufügen.
Auch Layer die direkt über den WebSuite :ref:`Client <client>` eingebunden werden, können mit gedruckt werden.
Ein Implementieren in den WebSuite Server ist nicht notwendig.

Bei der Druckfunktion wird der aktuelle Kartenausschnitt mit seinen Inhalten unverändert übernommen.
Dazu gehören: auf der Karte gezeichnete, beschriftete oder durch eine Abfrage markierte Objekte,
globale Einstellung wie die festgelegte :ref:`Rotation <turn>` und maßstabsbezogene Ebenendarstellungen,
sowie manuell festgelegte Transparenzen für einzelne Layer und Layergruppen.
Beim Aktivieren der Druckfunktion wechselt die Ansicht der GBD WebSuite.

.. figure:: ../../../screenshots/de/client-user/print3.png
  :align: center

Es öffnet sich auf der Karte ein Kartenrahmen mit dazugehörigem Dialogfenster.
Der Kartenrahmen stellt den zu druckenden Ausschnitt dar. Das Dialogfenster bietet die Konfigurationsmöglichkeiten dafür.

.. figure:: ../../../screenshots/de/client-user/print1.png
  :align: center

Hier können Sie die Ausrichtung der ``Druckvorlage`` und somit die Ausdehnung des Kartenrahmens konfigurieren.
Die Auswahl der ``Druckqualität`` kann unterschiedlich bereitgestellt werden.
Es ist möglich, dass vordefinierte Werte mit Titeln wie niedrig, mittel, hoch verknüpft sind, die dann zur Wahl stehen.
Alternativ ist es auch möglich, dass über ein Schieberegler der DPI-Wert frei gewählt werden kann.
In der Regel steht dann aber kein DPI-Wert von unter 70 DPI zur Verfügung, da sonst die Qualität des Drucks nicht gewährt wäre.
Unter ``Beschriftung`` können Sie einen Titel für den Druck vergeben.

Der Kartenrahmen kann beliebig verschoben werden.
Mit der linken Maustaste und dem Mausrad kann der entsprechende Ausschnitt und die Zoomstufe eingestellt werden.
Wenn Sie alles für den Druck vorbereitet haben, können Sie über das Drucksymbol |print| im Dialogfenster die druckfertige PDF generieren und öffnen.

.. |print| image:: ../../../images/baseline-print-24px.svg
    :width: 30em

.. können Sie über das Drucksymbol |print| den Export der druckfertigen PDF starten. Es öffnet sich ein Druckvorschaumodul. Hier wird Ihnen eine Vorschau des Drucks bereitgestellt. So kann der gewählte Ausschnitt nochmal kontrolliert werden.

 .. |print| image:: ../../../images/baseline-print-24px.svg
   :width: 30em
 .. |cancel| image:: ../../../images/baseline-cancel-24px.svg
   :width: 30em

.. .. figure:: ../../../screenshots/de/client-user/print_2.png
      :scale: 60%
      :align: center
