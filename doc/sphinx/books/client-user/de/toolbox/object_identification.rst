.. _object_identi:

Abfragen
========

Mit Hilfe des |info| ``Abfragen``-Werkzeugs können Objekte in der Karte selektiert werden,
derer Objekteigenschaften sich dann in einem Pop-up Fenster öffnen.

Wählen Sie das Werkzeug in der Werkzeugleiste aus. Sie erkennen das es aktiviert ist daran, dass sich der Mauszeiger verändert hat.
Jetzt können Sie ein Objekt in dem Kartenfenster wird mit der linken Maustaste auswählen.
Das gewählte Objekt wird markiert und es öffnet sich ein Pop-up Fenster, in dem die Objekteigenschaften abgelesen werden können.
Welche Objekte mit diesem Werkzeug selektiert werden können, wird durch die Auswahl im :ref:`Layer <map_element>` Menü definiert.
Es gibt verschiedene Möglichkeiten, die Auswahl genauer zu definieren.
Durch die Auswahl eines übergeordneten Layers, greift die Abfrage auf alle darunter liegenden Layer zu.

.. figure:: ../../../screenshots/de/client-user/mouseover_identification_1.png
  :align: center

Alternativ kann der unterste Layer in einer Layerstruktur gewählt werden, wodurch die Abfrage ausschließlich auf diesen Layer greift.

.. figure:: ../../../screenshots/de/client-user/mouseover_identification_2.png
  :align: center

Wenn Sie mehrere Objekte markiert haben, können Sie über |continue| und |back| durch die verschiedenen Objekteigenschaften navigieren.
Über das ``Hinzoomen``-Werkzeug können Sie die einzelnen Objekte fokussieren.
Unter |options| stehen weitere Funktionen zur Verfügung, bei denen die Auswahl der Objekte übernommen wird:

 * :ref:`Auswählen <selecting>`
 * :ref:`Räumliche Suche <spatial_searching>`
 * :ref:`Annotieren <measure>`

Über |cancel| schließen Sie das Pop-up Fenster. ( Sollen so triviale Sachen noch erwähnt werden, kann man eigentlich von jedem Nutzer erwarten, das Symbol mit einem Schließen zu verbinden oder?)


 .. |info| image:: ../../../images/gbd-icon-abfrage-01.svg
   :width: 30em
 .. |continue| image:: ../../../images/baseline-chevron_right-24px.svg
   :width: 30em
 .. |back| image:: ../../../images/baseline-keyboard_arrow_left-24px.svg
   :width: 30em
 .. |options| image:: ../../../images/round-settings-24px.svg
   :width: 30em
 .. |cancel| image:: ../../../images/baseline-close-24px.svg
   :width: 30em
