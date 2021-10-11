.. _cadastral_unit_searching:

Flurstücksuche
==============

In der Menüleiste |menu| findet man unter |cadastralunit| die ``Flurstücksuche``, welche auf amtliche ALKIS-Daten angewendet werden kann.
Das Modul und der Zugriff darauf kann individuell projekt- und benutzerspezifisch konfiguriert, Anfragen zum Einhalten des Datenschutzes geloggt werden.

Um einen Überblick zu erhalten, werden im folgenden Text die einzelnen Bestandteile genauer erläutert.

Starten der Flurstücksuche
--------------------------

Öffnet man die Flurstücksuche über die Menüleiste, ist als erstes die Suchmaske der Flurstücksuche zu sehen. Hier stehen verschiedene Parameter zur Auswahl die individuell konfiguriert werden können. So kann man Flurstücke anhand von Eigentümern, Adressen, Gemarkungen, Flächengrößen, Buchungsblatt- oder Flurstücknummern suchen.

Die Flurstücksuche wird über das Icon |search| ``Suche`` gestartet. Eine neue Anfrage kann per |new_search| ``Neue Anfrage`` gestartet werden. Dabei werden die Inhalte aller Felder gelöscht.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_1.png
  :align: center

Es gibt verschiedene Möglichkeiten zur Konfiguration der Flurstücksuche. Es müssen nicht alle Parameter definiert werden. So können Sie definieren ob die Suche nach einem speziellen Flurstück oder nach allen Flurstücken einer Straße erfolgen soll. Um nach einer Straße zu suchen, muss auch keine Gemarkung eingetragen werden. So ist die Suche auch bei großen Datenmengen performant möglich. Ob die Suche nach Straßen mit oder ohne Vorauswahl der Gemarkung stattfinden soll, kann ebenfalls konfiguriert werden. Bei der Suche anhand von Mindest- und Maximalfläche in Quadratmetern, können Kommastellen für das Eintragen von Zentimeter Werten verwendet werden.  Folgende Konfigurationen stehen zur Verfügung:

Optionale Konfigurationen zur Darstellung der Gemarkungsliste
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* keine Gemarkungsliste anzeigen
* die Gemarkungsliste anzeigen als eine Liste von Gemarkungsnamen
* die Gemarkungsliste anzeigen als eine Liste von Gemarkung- und Gemeindenamen in Klammern
* die Gemarkungsliste wird in einer Baumansicht angezeigt (erst Gemeinde, darunter eingerückt Gemarkung)

Optionale Konfigurationen zur Darstellung der Straßenliste
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* nur Straßennamen
* Straßenname und Gemeindenamen in Klammern:

Optionale Konfigurationen zur Straßensuche
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Die Eingabe "Mar" sucht am Anfang der Straßennamen und findet nur "Marktstraße"
* Die Eingabe "Mar" sucht überall in den Straßennamen und findet "Marktstraße" und "Neuer Markt"

.. note::
 Die Abfrage nach einer Buchungsblattnummer über das Buchungsblatt-Feld kann konfiguriert werden. So kann eine Plausibilität geprüft und frei definiert werden, welche Werte eingetragen werden, z.B. Zahlen oder Buchstaben, ob Bestimmte Werte am Anfang oder Ende oder grundsätzlich in dem Feld enthalten sein müssen.

Anzeigen der Ergebnisse
-----------------------

.. note::
 Für die Ausgabe der Ergebnisse kann die maximale Anzahl der Flurstücke konfiguriert werden.

Nach dem Anklicken des Suchbuttons |search| unter den Suchfeldern, werden die Ergebnisse in der Karte markiert und unter |results| ``Ergebnisse`` aufgelistet.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_2.png
  :align: center

Durch Klicken auf die Flurstücksbezeichnung gelangen Sie zu den Flurstücksinformationen. Diese können individuell konfiguriert werden und bestehen in diesem Beispiel aus den Abschnitten Basisdaten, Lage, Gebäudenachweis, Buchungssatz und Nutzung. Falls Sie die Flurstücksinformationen wieder schließen möchten, wählen Sie den gewünschten Menüpunkt der Flurstücksuche. Über die Symbole |add| und |delete| rechts neben der Flurstücksbezeichnung, können einzelne Flurstücke in eine Ablage gelegt werden. Diese kann für einen späteren Zugriff gespeichert werden. Die darin befindlichen Flurstücke können ausgedruckt oder in eine CSV Tabelle exportiert werden.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_4.png
  :align: center

.. Die gewonnenen Ergebnisse können durch klicken des neben dem Objekt stehenden |fokus| Symbol fokussiert werden. Außerdem ist es möglich über das Icon |add| ein Objekt der Ablage hinzu zu fügen oder über das |delete| Icon, ein Objekt wieder aus der Ablage zu entfernen. Oder über das |addall| Icon ebenfalls in der Leiste am unteren Fensterrand ``Alle zur Ablage`` hinzufügen.  So können Sie in der |tab| ``Ablage`` , welche ebenfalls am unteren Fensterrand der ``Flurstücksuche`` zu finden ist, eine Sammlung gesuchter Flurstücke anlegen und diese |save| ``Speichern``, |load| ``Laden``, |csv| als CSV-Datei exportieren oder |print| ``Drucken``.

Die nachfolgende Tabelle bildet eine Übersicht der vorhandenen Schaltflächen und deren Funktion im Ablagefenster ab.

+------------------------+--------------------------------------------------------------------------------------+
| **Icon**               | **Funktion**                                                                         |
+------------------------+--------------------------------------------------------------------------------------+
| |fokus|                | Zoomen auf das entsprechende Flurstück                                               |
+------------------------+--------------------------------------------------------------------------------------+
| |add|                  | ein Objekt zur Ablage hinzufügen                                                     |
+------------------------+--------------------------------------------------------------------------------------+
| |delete|               | ein Objekt aus der Ablage entfernen                                                  |
+------------------------+--------------------------------------------------------------------------------------+
| |addall|               | alle gewählten Objekte zur Ablage hinzufügen                                         |
+------------------------+--------------------------------------------------------------------------------------+
| |tab|                  | Ablage der ausgewählten Flurstücke                                                   |
+------------------------+--------------------------------------------------------------------------------------+
| |save|                 | Speichern der in der Ablage befindlichen Flurstücke                                  |
+------------------------+--------------------------------------------------------------------------------------+
| |load|                 | Öffnen von zuvor gespeicherten Ablagen von Flurstücken                               |
+------------------------+--------------------------------------------------------------------------------------+
| |csv|                  | Die in der Ablage befindlichen Flurstücke werden als CSV exportiert                  |
+------------------------+--------------------------------------------------------------------------------------+
| |print|                | Drucken der in der Ablage befindlichen Flurstücke, Ausgabe im Format PDF             |
+------------------------+--------------------------------------------------------------------------------------+
| |delete_shelf|         | Leeren der Ablage                                                                    |
+------------------------+--------------------------------------------------------------------------------------+

.. Wenn Sie ein einzelnes Objekt angewählt haben, gibt es zusätzlich Funktionen die nur dann möglich sind. Sie können zum einen wieder über das Icon |add| ein Objekt der Ablage hinzu zu fügen oder über das |delete| Icon, ein Objekt wieder aus der Ablage entfernen. Zusätzlich können die Informationen des Objektes gedruckt werden oder weitere Funktionen, ähnlich wie beim ``Auswahl``-Menü, gewählt werden. Die Erklärung für die Funktionen ``Räumliche Suche`` und ``Markieren und Messen`` entnehmen Sie bitte dem jeweiligen Punkt in dieser Hilfe. Über ``Auswahl`` kehren Sie wieder zum ursprünglichen ``Auswahl``-Werkzeug zurück. Über |fokus| ``Hinzoomen`` fokussieren Sie das gewünschte Objekt.

Exportieren und Drucken
-----------------------

Exportieren als CSV
~~~~~~~~~~~~~~~~~~~

Die Flurstückinformationen bestehen aus unterschiedlichen Teilbereichen. Beim Export in das CSV Format können Sie eine Auswahl treffen und dann auf den Button ``Exportieren`` klicken.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_area_csv.png
  :align: center

.. note::
   Es kann passieren das bei der Auswahl aller Daten in der Ergebnistabelle einige Flurstücke mehrfach auftauchen. Das ist u.a. dann der Fall, wenn ein Flurstück mehrere Eigentümer hat.

Drucken der Flurstückinformationen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Über das |print| Symbol können Sie die ausgewählten Flurstückinformationen drucken. Der Inhalt des Ausdrucks kann individuell über ein Template vorbereitet werden. Dieses kann auch einen Kartendarstellung des jeweiligen Flurstücks beinhalten.

Arbeiten mit der Ablage
-----------------------

Die |tab| Ablage ist ein Bereich, der genutzt werden kann, um bestimmte Flurstücke einer Suchanfrage und/oder Ergebnisse mehrerer Suchanfragen abzulegen. Man kann es als eine Art Sammelstelle verstehen, deren Inhalt letztlich für die Beauskunftung verwendet werden kann.

Speichern
~~~~~~~~~

Über das |save| Symbol können Sie ausgewählte Flurstücke oder Flurstücke der Ablage in einer benutzerspefischen Ablage speichern. Vergeben Sie einen gewünschten Namen und speichern Sie durch Klicken auf das Häckchen.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_print_save.png
  :align: center

Laden
~~~~~

Über das |load| Symbol kann eine benutzerspefische Ablage wieder geladen werden. Wählen Sie einen Ablagenamen und laden Sie diese durch Klicken auf das Häckchen.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_print_load.png
  :align: center

.. note::
  Das Speichern benutzerspefischer Ablagen geschieht in einer SQLite Datenbank.

Flurstücke in der Karte suchen
------------------------------

Mit Hilfe des Werkzeuges |spatial_search| ``Flurstücke über räumliche Suche finden`` ist es möglich, Flurstücke durch das Zeichnen von Geometrien zu ermitteln. Es öffnet sich in der Toolbar die Leiste für die Räumliche Suche, mit dem Verweis, dass nun in den Flurstücken gesucht wird.

**Beispiel: Auswahl Flurstücke durch Linie**

Wählen Sie das Tool Linie aus. Zeichnen Sie nun parallel zu einer Straße eine Linie und schließen Sie diese mit einem Doppelklick ab. Anschließend werden Ihnen die Flurstücke in den Ergebnissen aufgelistet, welche von der Linie geschnitten werden. Auf dem Screenshot unten sieht man ein mögliches Ergebnis.

 .. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_area_search.png
   :align: center

Flurstücke in der Karte wählen
------------------------------

Das Werkzeug |select| ``Einzelne Flurstücke in der Karte wählen`` ermöglicht die Selektierung einzelner Flurstücke. Sobald das Werkzeug aktiviert ist, können Sie per Mausklick gewünschte Flurstücke aussuchen, welche dann in der Ablage aufgelistet werden.

 .. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_5.png
   :align: center

 .. |menu| image:: ../../../images/baseline-menu-24px.svg
   :width: 30em
 .. |cadastralunit| image:: ../../../images/gbd-icon-flurstuecksuche-01.svg
   :width: 30em
 .. |results| image:: ../../../images/baseline-menu-24px.svg
   :width: 30em
 .. |tab| image:: ../../../images/sharp-bookmark_border-24px.svg
   :width: 30em
 .. |fokus| image:: ../../../images/sharp-center_focus_weak-24px.svg
   :width: 30em
 .. |add| image:: ../../../images/sharp-control_point-24px.svg
   :width: 30em
 .. |addall| image:: ../../../images/gbd-icon-alle-ablage-01.svg
   :width: 30em
 .. |delete| image:: ../../../images/sharp-remove_circle_outline-24px.svg
   :width: 30em
 .. |save| image:: ../../../images/sharp-save-24px.svg
   :width: 30em
 .. |load| image:: ../../../images/gbd-icon-ablage-oeffnen-01.svg
   :width: 30em
 .. |csv| image:: ../../../images/sharp-grid_on-24px.svg
   :width: 30em
 .. |print| image:: ../../../images/baseline-print-24px.svg
   :width: 30em
 .. |search| image:: ../../../images/baseline-search-24px.svg
   :width: 30em
 .. |select| image:: ../../../images/gbd-icon-auswahl-01.svg
   :width: 30em
 .. |spatial_search| image:: ../../../images/gbd-icon-raeumliche-suche-01.svg
   :width: 30em
 .. |delete_shelf| image:: ../../../images/sharp-delete_forever-24px.svg
   :width: 30em
 .. |new_search|  image:: ../../../images/baseline-delete_sweep-24px.svg
   :width: 30em

.. Protokollierter Zugang zu Personendaten
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. Zum einbehalten der Datenschutzrechte gibt es folgendes Werkzeug in der Flurstücksuche. Wenn jemand nach personenbezogenen Daten wie Vorname und Name sucht, muss das Häckchen bei  ``Zugang zu Personendaten`` gesetzt werden. Es öffnet sich ein Fenster in dem ein firmeninternes Aktenzeichen vergeben werden muss. Dieses Aktenzeichen verifiziert den Erhalt der personenbezogenen Daten. Außerdem wird jede Anfrag in einer PostGIS Datei abgelegt, sodass jede Anfrage protokolliert und somit kontrolliert werden kann.

.. .. figure:: ../../../screenshots/de/client-user/cadastral_unit_search_data_rights.png
