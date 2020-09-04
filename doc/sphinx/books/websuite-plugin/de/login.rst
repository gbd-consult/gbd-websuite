Anmelden
========

Bevor Sie das GBD WebSuite Manager Plugin nutzen können müssen Sie sich auf dem GBD WebSuite Server anmelden.
Dafür benötigt man eine Login-Datei im JSON-Format, in welcher der Nutzername mit Passwort und die Serveradresse abgelegt sind.
Der Aufbau der JSON-Datei entnehmen Sie bitte folgendem Screenshot:

.. figure:: ../../../screenshots/de/websuite-plugin/loginexample.png
  :align: center

Zwischen den grünen Anführungszeichen müssen die Werte für Nutzername, Passwort und Serveradresse eingetragen werden.
Wenn diese Datei angelegt ist, kann man über den |browse| ``Durchsuchen``-Button im GBD WebSuite Plugin, den Dateibrowser öffnen und die JSON-Datei auswählen.
Wenn die Logindaten richtig und auf dem Server als registrierter Nutzer eingetragen sind, verbindet das Plugin sich automatisch.

Außerdem ist es möglich eine automatische Anmeldung einzurichten, sodass die Logindatei nicht jedes mal manuell ausgewählt werden muss.
Dazu muss unter dem Pfad '.local/share/QGIS/QGIS3/profiles/default' der Ordner 'GBD_WebSuite' angelegt werden.
Kopieren Sie dann in '.local/share/QGIS/QGIS3/profiles/default/GBD_WebSuite' die Logindatei.
Wichtig ist zu beachten, das die Logindatei als 'conf.json' bennant ist.

.. |browse| image:: ../../../images/more_horiz-24px.svg
  :width: 30em
.. |trash| image:: ../../../images/mActionTrash.png
  :width: 1em
