Flurstückssuche
===============

In der Menüleiste |menu| findet man unter dem Icon die |cadastralunit| ``Flurstückssuche`` mit amtliche ALKIS-Daten. Um einen genauen Überblick zu erhalten, werden im folgenden Text die einzelnen Bestandteile genauer erläutert.

Starten der Flurstückssuche
---------------------------

Öffnet man die Flurstückssuche über die Menüleiste, öffnet sich als erstes das oben zu sehende Fenster. Hier stehen verschiedene Suchfelder zur Auswahl, welche auf verschiedene Weisen verwendet werden können. Zum einen ist es möglich nach einem speziellen Flurstück zu suchen indem die individuellen Daten eingegeben werden. Zum anderen ist es möglich diese Felder wie eine Art Filter zu verwenden.

Die Flurstückssuche wird über das Icon |search| gestartet.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_1.png

+------------------------+---------------------------------+
| **Feld**               | **Beispiel**                    |
+------------------------+---------------------------------+
| Vorname                | Max                             |
+------------------------+---------------------------------+
| Nachname               | Mustermann                      |
+------------------------+---------------------------------+
| Gemarkung              | Albshausen (Rauschenberg)       |
+------------------------+---------------------------------+
| Straße                 | Am Bingel                       |
+------------------------+---------------------------------+
| Nummer                 | 2b                              |
+------------------------+---------------------------------+
| Flur-Zähler/Nenner     | 6-30/6                          |
+------------------------+---------------------------------+
| Fläche                 | 30 bis 50 m²                    |
+------------------------+---------------------------------+
| Buchungsblattnummer    | 0013658                         |
+------------------------+---------------------------------+

.. note::
    Bei der Suche nach ``Flur-Zähler/Nenner`` ist explizit auf die Syntax zu achten:
    **Flur-Zähler/Nenner**.
    Optional kann auch nach den einzelnen Bestandteilen *Flur*, *Zähler* oder *Nenner* gesucht werden.


Anzeigen der Ergebnisse
-----------------------

.. note::
 Bei der Ausgabe der Ergebnisse ist zu beachten, dass die Ergebnissspalte maximal 100 Ergebnissen ausgibt. Grund: Es wird eine weitere Spezifizierung erwartet.

Nach dem Anklicken des Suchbuttons |search| unten links, werden die Ergenisse in der Karte und im Ergebnisfenster |results| ``Ergebnisse`` der Sidebar angezeigt.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_2.png

Über den Klick auf den Ergebnistext *Flurstück* eines Eintrags gelangen Sie zu den Flurstücksinformationen. Diese bestehen aus den Abschnitten Basisdaten, Lage, Gebäudenachweis, Buchungssatz und Nutzung.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_4.png

.. Die gewonnenen Ergebnisse können durch klicken des neben dem Objekt stehenden |fokus| Symbol fokusiert werden. Außerdem ist es möglich über das Icon |add| ein Objekt der Ablage hinzu zu fügen oder über das |delete| Icon, ein Objekt wieder aus der Ablage zu entfernen. Oder über das |addall| Icon ebenfalls in der Leiste am unteren Fensterrand ``Alle zur Ablage`` hinzufügen.  So können Sie in der |tab| ``Ablage`` , welche ebenfalls am unteren Fensterrand der ``Flurstückssuche`` zu finden ist, eine Sammlung gesuchter Flurstücke anlegen und diese |save| ``Speichern``, |load| ``Laden``, |csv| als CSV-Datei exportieren oder |print| ``Drucken``.

Arbeiten mit der Ablage
-----------------------

.. .. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_3.png
  :align: center

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
| |load|                 | Öffnen von zurvor gespeicherten Ablagen von Flurstücken                              |
+------------------------+--------------------------------------------------------------------------------------+
| |csv|                  | Die in der Ablage befindlichen Flurstücke werden als CSV exportiert                  |
+------------------------+--------------------------------------------------------------------------------------+
| |print|                | Drucken der in der Ablage befindlichen Flurstücke, Ausgabe im Format PDF             |
+------------------------+--------------------------------------------------------------------------------------+

.. Wenn Sie ein einzelnes Objekt angewählt haben, gibt es zusätzlich Funktionen die nur dann möglich sind. Sie können zum einen wieder über das Icon |add| ein Objekt der Ablage hinzu zu fügen oder über das |delete| Icon, ein Objekt wieder aus der Ablage entfernen. Zusätzlich können die Informationen des Objektes gedruckt werden oder weitere Funktionen, ähnlich wie beim ``Auswahl``-Menü, gewählt werden. Die Erklärung für die Funktionen ``Räumliche Suche`` und ``Markieren und Messen`` entnehmen Sie bitte dem jeweiligen Punkt in dieser Hilfe. Über ``Auswahl`` kehren Sie wieder zum ursprünglichen ``Auswahl``-Werkzeug zurück. Über |fokus| ``Hinzoomen`` fokusieren Sie das gewünschte Objekt.


Arbeiten mit der räumlichen Suche
---------------------------------

Beschreibung folgt.


Auswahl eines Flurstücks in der Karte
-------------------------------------

Beschreibung folgt.


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



.. .. note::
    Auf Wunsch kann diese Funktion für manche oder alle Nutzer deaktiviert werden. Außerdem ist eine andere Anordnung möglich.
