.. _editing:

Editieren
=========

In der Menüleiste |menu| ist ebenfalls der Punkt |edit| ``Editieren`` zu finden.
Unter diesem Menüpunkt ist es möglich, vorhandene Objekte auszuwählen und zu editieren, sowie neue Objekte hinzuzufügen.
Wenn Sie die Funktion in der Menüleiste auswählen, öffnet sich eine Liste der vorhandenen Vektorlayer.
In unserem Beispiel sind die beiden Layer "Restaurants" und "Gewässer" implementiert.

.. figure:: ../../../screenshots/de/client-user/editing_1.png
 :align: center

Durch das Anwählen eines Layers, öffnet sich die Liste mit den dazugehörigen Objekten.
Prominent oberhalb der Liste angeordnet, befindet sich die Suchleiste.
Darüber können Sie nach Objekten suchen.

.. figure:: ../../../screenshots/de/client-user/editing_2.png
 :align: center

Über |fokus| kann das jeweilige Objekt fokussiert werden.
Durch ein Anwählen eines Objekts, öffnen sich die dazugehörigen Objekteigenschaften.
Zusätzlich befinden sich hilfreiche Werkzeuge unterhalb der Liste.
Mit dem |select_editing| ``Bearbeiten`` Werkzeug ist es möglich, Objekte in der Karte auszuwählen.
Durch ein einfaches Anklicken eines Objektes, öffnen sich die dazugehörigen Objekteigenschaften.
Jetzt kann durch ein erneutes Anklicken des Objekts, mit halten der linken Maustaste nach dem Anklicken, das Objekt neu platziert werden.
Alternativ kann die Platzierung über die X und Y Koordinaten angepasst werden.
Über das Werkzeug ``Zeichnen`` |new_editing| ist es möglich, je nachdem welcher Layertyp vorhanden ist, neue Objekte in Form von Punkten, Linien oder Polygonen hinzuzufügen.
Nach dem Zeichnen der Geometrie, öffnen sich die Objekteigenschaften. Hier können Werte für die verschieden Attribute eingetragen werden.
Es können restriktive Felder konfiguriert sein, die ein Abspeichern des Objekts erst ermöglichen, wenn ein Wert für diese eingetragen wurde.
Mit der Funktion |add| ``Hinzufügen`` können Sie neue Objekte durch die Eingabe von Koordinaten erstellen.
Außerdem können Sie über |cancel| ``Beenden`` zurück zur Layeransicht navigieren.

Wenn ein Objekt ausgewählt oder ein neues Objekt erstellt wurde, öffnen sich die Objekteigenschaften.
Hier kann sich ein Überblick, über sämtliche Attributwerte verschafft werden.

.. figure:: ../../../screenshots/de/client-user/editing_3.png
  :align: center

In unserem Beispiel können "ID", "Feature Klasse", "Name" und "X-Y Koordinaten" abgelesen werden.
Die Attributwerte für "ID" und "Feature Klasse" sind so konfiguriert, dass sie nicht verändert werden können.
"Name" sowie "X-Y Koordinaten" (und somit die Position) können hingegen angepasst werden.
Außerdem könnte eine Bereitstellung von gewissen Attributen, für verifizierte Nutzer, konfiguriert werden.
Dies würde dazu führen, dass nicht jeder Nutzer Einsicht auf alle Attribute hat.
Ebenfalls können vorgegebene Wertebereich definiert sein, in denen die Eingabe liegen muss.
Außerdem ist es möglich, dass nur ein Datum eingetragen werden kann, weil das Feld als Datumsfeld definiert ist.
Dies kann für jedes GBD WebSuite Projekt individuell konfiguriert sein. Über das Menü |settings| ``Aufgaben`` können weitere Funktionen ausgewählt werden.
Hier steht die Funktion ``Hinzoomen`` zur Verfügung. Über |cancel| ist es möglich ins vorherige Menü zu navigieren.

Außerdem ist es möglich eingebundene Web-Formulare anzupassen.
Darüber können zum Beispiel Umfragen, Antrags-, Anmelde- und Bestellformulare bereitgestellt werden.
Diese Formulare können Sie dann auf Ihrer Internet- oder Intranetseite veröffentlichen oder per E-Mail und sozialen Medien teilen.

.. note::
 Falls diese Funktion nicht zur Verfügung stehen soll, ist es möglich sie zu deaktivieren.
 Es wäre zum Beispiel möglich, dass die Funktion ``Zeichnen`` nur Linien Zeichnen ermöglicht, aber keine Flächen.
 Diese Funktion ganz zu deaktivieren, wäre auch möglich.

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
