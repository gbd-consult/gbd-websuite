.. _editing:

Editieren
=========

Unter dem Menüpunkt |edit| :guilabel:`Editieren` ist es möglich, vorhandene Objekte zu editieren und neue Objekte hinzuzufügen.
Wenn Sie die Funktion in der Menüleiste auswählen, öffnet sich eine Liste der vorhandenen Vektorlayer.
In unserem Beispiel sind die beiden Layer "Restaurants" und "Gewässer" implementiert.

.. figure:: ../../../screenshots/de/client-user/editing_1.png
 :align: center

Durch das Anwählen eines Layers, öffnet sich die Liste mit den vorhandenen Objekten dieses Layers.
Oberhalb der Liste angeordnet, befindet sich die Suchleiste.
Darüber können Sie nach Objekten suchen.
Über |fokus| kann das jeweilige Objekt fokussiert werden.

.. figure:: ../../../screenshots/de/client-user/editing_2.png
 :align: center

Zusätzlich befinden sich hilfreiche Werkzeuge unterhalb der Liste.
Mit dem |select_editing| ``Bearbeiten`` Werkzeug ist es möglich, Objekte in der Karte auszuwählen.
Durch ein einfaches Anklicken eines Objektes, öffnen sich die dazugehörigen Objekteigenschaften.
Durch ein erneutes Anklicken des Objekts, mit halten der linken Maustaste, kann das Objekt neu platziert werden.
Alternativ kann die Platzierung über die :command:`X` und :command:`Y` Koordinaten angepasst werden.
Über das Werkzeug |new_editing| ``Zeichnen`` ist es möglich, je nachdem welcher Layertyp vorhanden ist, neue Objekte in Form von Punkten, Linien oder Polygonen hinzuzufügen.
Nach dem Zeichnen der Geometrie, öffnen sich die Objekteigenschaften. Falls dem Objekt Dateien oder Bilder angehangen sind, können diese hier geladen werden.
Hier kann ein Überblick über die vorhandenen Attributfelder gewonnen, sowie Attributwerte eingetragen werden.

.. figure:: ../../../screenshots/de/client-user/editing_3.png
  :align: center

.. tip::
 Es können restriktive Felder konfiguriert sein, die ein Abspeichern des Objekts erst ermöglichen, wenn ein Wert für diese eingetragen wurde.

Mit der Funktion |add| ``Hinzufügen`` können Sie neue Objekte durch die Eingabe von Koordinaten erstellen.
Außerdem können Sie über |cancel| ``Beenden`` zurück zur Layeransicht navigieren.


.. note::
 Attributfelder können durch primitive wie auch komplexe Datentypen definiert sein.
 Primitive Datentypen entsprechen Datenbank-Typen. Es werden folgende Datentypen unterstützt:

 * string
 * integer
 * float
 * boolean
 * date
 * datetime

 Komplexe Datentypen werden als primitive Datenbank-Typen gespeichert, haben jedoch eine andere Semantik. Zu komplexen Datentypen gehören

 * measurement: numerischer Wert mit einer Maßeinheit, wie 2 m oder 4 ha
 * money: monetärer Wert wie 12.34
 * currency: monetärer Wert mit Währung, wie 12.34 EUR
 * crs KBS Wert wie 3857
 * extent: 4 reelle Zahlen die eine räumliche BoundingBox beschreiben

 Für jedes GBD WebSuite Projekt können die Attributfelder individuell konfiguriert sein.
 Sie können Fest- und Defaultwerte zugeordnet haben oder als Datumsfeld definiert sein.
 In unserem Beispiel können :command:`ID`, :command:`Feature Klasse`, :command:`Name` und :command:`X-Y Koordinaten` abgelesen werden.
 Die Attributwerte für :command:`ID` und :command:`Feature Klasse` sind so konfiguriert, dass sie nicht verändert werden können.
 :command:`Name` sowie :command:`X-Y Koordinaten` (und somit die Position) können hingegen angepasst werden.
 Ebenfalls kann ein Wertebereich definiert sein, in dem die eingegebenen Werte liegen müssen.

 Außerdem ist es möglich eingebundene Web-Formulare anzupassen.
 Darüber können zum Beispiel Umfragen, Antrags-, Anmelde- und Bestellformulare bereitgestellt werden.
 Diese Formulare können Sie dann auf Ihrer Internet- oder Intranetseite veröffentlichen oder per E-Mail und sozialen Medien teilen.


 .. |add| image:: ../../../images/sharp-control_point-24px.svg
   :width: 30em
 .. |menu| image:: ../../../images/baseline-menu-24px.svg
   :width: 30em
 .. |edit| image:: ../../../images/sharp-edit-24px.svg
   :width: 30em
 .. |select_editing| image:: ../../../images/cursor.svg
   :width: 30em
 .. |new_editing| image:: ../../../images/draw_black_24dp.svg
   :width: 30em
 .. |delete_editing| image:: ../../../images/baseline-delete-24px.svg
   :width: 30em
 .. |cancel| image:: ../../../images/baseline-close-24px.svg
   :width: 30em
 .. |fokus| image:: ../../../images/sharp-center_focus_weak-24px.svg
   :width: 30em
 .. |settings| image:: ../../../images/round-settings-24px.svg
   :width: 30em
