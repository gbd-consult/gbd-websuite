.. _dimensioning:

Bemaßung
========

Das |dimensions| :guilabel:`Bemaßung`-Werkzeug ermöglicht die Erfassung von Streckenlängen, durch das Zeichnen von Linien. Ähnlich einer technischen Bemaßung.
Die gezeichneten Linien werden automatisch mit den jeweiligen Streckenlängen beschriftet. Zusätzlich kann eine individuelle Beschriftung angehangen werden.

**Bedienung:**

Nach Aktivierung des Werkzeugs, öffnet sich unter der Werkzeugleiste ein Feld mit folgenden Optionen:

 .. figure:: ../../../screenshots/de/client-user/dimensions_menu.png
  :align: center

.. table::
 :align: center

 +------------------------+-------------------------------------------------------------------------------------------------------------+
 |      **Symbol** | **Beschreibung**                                                                                                   |
 +-----------------+--------------------------------------------------------------------------------------------------------------------+
 |         |arrow| |   ``Bemaßung auswählen und bearbeiten``                                                                            |
 +-----------------+--------------------------------------------------------------------------------------------------------------------+
 |* |1| Mit Hilfe der gelb markierten Stützpunkte kann die Bemaßung versetzt werden.                                                    |
 |* |2| Durch ein Klicken auf den grünen Hilfspunkt, kann sowohl die Bemaßung als auch die Beschriftung auf der Linie verschoben werden.|
 +-----------------+--------------------------------------------------------------------------------------------------------------------+
 |         |line|  |   ``Bemaßung zeichnen``                                                                                            |
 +-----------------+--------------------------------------------------------------------------------------------------------------------+
 |* Klicken Sie einmal um einen neuen Stützpunkt zu setzen                                                                              |
 |* Klicken Sie zweimal um das Zeichnen einer Linie zu beenden                                                                          |
 |* Halten Sie die Mausgedrückt um den Kartenausschnitt zu verschieben.                                                                 |
 +-----------------+--------------------------------------------------------------------------------------------------------------------+
 |        |delete| |   ``ausgewählte Bemaßung löschen``                                                                                 |
 +-----------------+--------------------------------------------------------------------------------------------------------------------+
 |* Durch einen Klick auf das Icon werden die ausgewählten Bemaßungen gelöscht.                                                         |
 +-----------------+--------------------------------------------------------------------------------------------------------------------+
 |        |cancel| |   ``Beenden``                                                                                                      |
 +-----------------+--------------------------------------------------------------------------------------------------------------------+
 |* Durch einen Klick auf das Icon wird das Werkzeug deaktiviert.                                                                       |
 +-----------------+--------------------------------------------------------------------------------------------------------------------+

.. * |arrow| ``Bemaßung auswählen und bearbeiten``

..  * |1| Mit Hilfe der gelb markierten Stützpunkte kann die Bemaßung versetzt werden.
..  * |2| Durch ein Klicken auf den grünen Hilfspunkt, kann sowohl die Bemaßung als auch die Beschriftung auf der Linie verschoben werden.

.. * |line| ``Bemaßung zeichnen``

..  * Klicken Sie einmal um einen neuen Stützpunkt zu setzen
..  * Klicken Sie zweimal um das Zeichnen einer Linie zu beenden
..  * Halten Sie die Mausgedrückt um den Kartenausschnitt zu verschieben

.. * |delete| ``ausgewählte Bemaßung löschen``

..  * Durch einen Klick auf das Icon werden die ausgewählten Bemaßungen gelöscht.

.. * |cancel| ``Beenden``

..  * Durch einen Klick auf das Icon wird das Werkzeug deaktiviert.


.. figure:: ../../../screenshots/de/client-user/dimensions1.png
 :align: center

Alle erstellten Bemaßungen werden in der Ablage des Menüpunkts :ref:`Bemaßungen <dimensions>` aufgelistet.
Dieser Menüpunkt öffnet sich automatisch, wenn das |dimensions| :guilabel:`Bemaßung`-Werkzeug aktiviert wird.
Durch das Anklicken einer Bemaßung, kann eine individuelle :command:`Beschriftung` vergeben werden.

.. figure:: ../../../screenshots/de/client-user/dimensions_annotate.png
 :align: center

Am unteren Menüfensterrand befinden sich folgende Funktionen für diesen Menüpunkt:

.. table::
 :align: center

 +------------------------+------------------------------------------------------+
 | **Symbol**             | **Bedienelement**                                    |
 +------------------------+------------------------------------------------------+
 |      |load|            |   ``Auswahl laden``                                  |
 +------------------------+------------------------------------------------------+
 |     |save|             |   ``Auswahl speichern``                              |
 +------------------------+------------------------------------------------------+
 |    |delete_marking|    |   ``Auswahl löschen``                                |
 +------------------------+------------------------------------------------------+

.. admonition:: Konfigurationsmöglichkeiten

 Die Längeneinheit kann durch Konfigurationen angepasst werden.

 .. |dimensions| image:: ../../../images/gbd-icon-bemassung-02.svg
   :width: 30em
 .. |arrow| image:: ../../../images/cursor.svg
   :width: 30em
 .. |line| image:: ../../../images/dim_line.svg
   :width: 30em
 .. |cancel| image:: ../../../images/baseline-close-24px.svg
   :width: 30em
 .. |trash| image:: ../../../images/baseline-delete-24px.svg
   :width: 30em
 .. |1| image:: ../../../images/gws_digits-01.svg
   :width: 35em
 .. |2| image:: ../../../images/gws_digits-02.svg
   :width: 35em
 .. |save| image:: ../../../images/sharp-save-24px.svg
   :width: 30em
 .. |load| image:: ../../../images/ic_folder_open_24px.svg
   :width: 30em
 .. |delete_marking| image:: ../../../images/sharp-delete_forever-24px.svg
   :width: 30em
 .. |delete| image:: ../../../images/baseline-delete-24px.svg
   :width: 30em
