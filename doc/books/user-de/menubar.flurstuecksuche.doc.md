# Flurstücksuche :/user-de/menubar.flurstuecksuche

Über das Menü ![](gbd-icon-flurstuecksuche-01.svg) {title Flurstücksuche} kann eine Beauskunftung zu amtlichen Liegenschaftsdaten "ALKIS" stattfinden.

Öffnet man die Flurstücksuche über die Menüleiste, ist als erstes die Suchmaske für Flurstücke zu sehen. Hier stehen konfigurierbare Suchfelder zur Auswahl. So kann man Flurstücke z.B. nach Eigentümern, Adressen, Gemarkungen, Flächengrößen, Buchungsblatt- oder Flurstücknummern suchen.

Die Flurstücksuche wird über das Icon ![](baseline-search-24px.svg) {button Suche} gestartet. Eine neue Anfrage findet über das Icon ![](baseline-delete_sweep-24px.svg) {button Neue Anfrage} statt. Dabei werden die bisherigen Inhalte aller Felder gelöscht.

![](cadastral_unit_searching_1.png)

**Anzeigen der Ergebnisse**

Nach dem Anklicken des Icons ![](baseline-search-24px.svg) {button Suche} unter den Suchfeldern, werden die Ergebnisse in der Karte markiert und unter |results| ``Ergebnisse`` aufgelistet.

![](cadastral_unit_searching_2.png)

Durch Klicken auf die Flurstücksbezeichnung gelangen Sie zu den Flurstücksinformationen. Diese können individuell konfiguriert werden und bestehen in diesem Beispiel aus den Abschnitten Basisdaten, Lage, Gebäudenachweis, Buchungssatz und Nutzung. Falls Sie die Flurstücksinformationen wieder schließen möchten, wählen Sie den gewünschten Menüpunkt der Flurstücksuche. Über die Symbole ![](sharp-control_point-24px.svg) {button Hinzufügen} und ![](sharp-remove_circle_outline-24px.svg) {button Löschen} rechts neben der Flurstücksbezeichnung, können einzelne Flurstücke in eine Ablage gelegt werden. Diese kann für einen späteren Zugriff gespeichert werden. Die darin befindlichen Flurstücke können ausgedruckt oder in eine CSV Tabelle exportiert werden.

![](cadastral_unit_searching_4.png)

Die gewonnenen Ergebnisse können durch klicken des neben dem Objekt stehenden ![](sharp-center_focus_weak-24px.svg) {button Hinzoomen} Symbol fokussiert werden. Außerdem ist es möglich über das Icon |add| ein Objekt der Ablage hinzu zu fügen oder über das |delete| Icon, ein Objekt wieder aus der Ablage zu entfernen. Oder über das |addall| Icon ebenfalls in der Leiste am unteren Fensterrand ``Alle zur Ablage`` hinzufügen.  So können Sie in der |tab| ``Ablage`` , welche ebenfalls am unteren Fensterrand der ``Flurstücksuche`` zu finden ist, eine Sammlung gesuchter Flurstücke anlegen und diese ![](save.svg) {button Speichern}, ![](load.svg) {button Laden}, |csv| als CSV-Datei exportieren oder ![](baseline-print-24px.svg) {button Drucken}.

Die nachfolgende Tabelle bildet eine Übersicht der vorhandenen Schaltflächen und deren Funktion im Ablagefenster ab.

|Symbol          				| Funktion                                                                     		|
|-----------------------------------------------|---------------------------------------------------------------------------------------|
| ![](sharp-center_focus_weak-24px.svg)		| Zoomen auf das entsprechende Flurstück                                        	|
| ![](sharp-control_point-24px.svg)	        | ein Objekt zur Ablage hinzufügen                                                      |
| ![](sharp-remove_circle_outline-24px.svg)     | ein Objekt aus der Ablage entfernen                                                   |
| ![](gbd-icon-alle-ablage-01.svg)	        | alle gewählten Objekte zur Ablage hinzufügen                                          |
| ![](sharp-bookmark_border-24px.svg)	        | Ablage der ausgewählten Flurstücke                                                    |
| ![](sharp-save-24px.svg)	                | Speichern der in der Ablage befindlichen Flurstücke                                   |
| ![](gbd-icon-ablage-oeffnen-01.svg)	   	| Öffnen von zuvor gespeicherten Ablagen von Flurstücken                                |
| ![](sharp-grid_on-24px.svg)	                | Die in der Ablage befindlichen Flurstücke werden als CSV exportiert                   |
| ![](baseline-print-24px.svg)	                | Drucken der in der Ablage befindlichen Flurstücke, Ausgabe im Format PDF              |
| ![](sharp-delete_forever-24px.svg)	        | Leeren der Ablage                                                                     |

 Die Erklärung für die Funktionen ``Räumliche Suche`` und ``Markieren und Messen`` entnehmen Sie bitte dem jeweiligen Punkt in dieser Hilfe. Über ``Auswahl`` kehren Sie wieder zum ursprünglichen ``Auswahl``-Werkzeug zurück.


TODO: BELLOW!!!
## Exportieren und Drucken**

**Exportieren als CSV**

Die Flurstückinformationen bestehen aus unterschiedlichen Teilbereichen. Beim Export in das CSV Format können Sie eine Auswahl treffen und dann auf den Button ``Exportieren`` klicken.

![](cadastral_unit_searching_area_csv.png)

%info
   Es kann passieren das bei der Auswahl aller Daten in der Ergebnistabelle einige Flurstücke mehrfach auftauchen. Das ist u.a. dann der Fall, wenn ein Flurstück mehrere Eigentümer hat.
%end

**Drucken der Flurstückinformationen**

Über das |print| Symbol können Sie die ausgewählten Flurstückinformationen drucken. Der Inhalt des Ausdrucks kann individuell über ein Template vorbereitet werden. Dieses kann auch einen Kartendarstellung des jeweiligen Flurstücks beinhalten.

## Arbeiten mit der Ablage

Die |tab| Ablage ist ein Bereich, der genutzt werden kann, um bestimmte Flurstücke einer Suchanfrage und/oder Ergebnisse mehrerer Suchanfragen abzulegen. Man kann es als eine Art Sammelstelle verstehen, deren Inhalt letztlich für die Beauskunftung verwendet werden kann.

**Speichern**

Über das |save| Symbol können Sie ausgewählte Flurstücke oder Flurstücke der Ablage in einer benutzerspefischen Ablage speichern. Vergeben Sie einen gewünschten Namen und speichern Sie durch Klicken auf das Häckchen.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_print_save.png
  :align: center

**Laden**

Über das |load| Symbol kann eine benutzerspefische Ablage wieder geladen werden. Wählen Sie einen Ablagenamen und laden Sie diese durch Klicken auf das Häckchen.

.. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_print_load.png
  :align: center

.. note::
  Das Speichern benutzerspefischer Ablagen geschieht in einer SQLite Datenbank.

## Flurstücke in der Karte suchen

Mit Hilfe des Werkzeuges |spatial_search| ``Flurstücke über räumliche Suche finden`` ist es möglich, Flurstücke durch das Zeichnen von Geometrien zu ermitteln. Es öffnet sich in der Toolbar die Leiste für die Räumliche Suche, mit dem Verweis, dass nun in den Flurstücken gesucht wird.

**Beispiel: Auswahl Flurstücke durch Linie**

Wählen Sie das Tool Linie aus. Zeichnen Sie nun parallel zu einer Straße eine Linie und schließen Sie diese mit einem Doppelklick ab. Anschließend werden Ihnen die Flurstücke in den Ergebnissen aufgelistet, welche von der Linie geschnitten werden. Auf dem Screenshot unten sieht man ein mögliches Ergebnis.

 .. figure:: ../../../screenshots/de/client-user/cadastral_unit_searching_area_search.png
   :align: center

## Flurstücke in der Karte wählen
------------------------------

Das Werkzeug |select| ``Einzelne Flurstücke in der Karte wählen`` ermöglicht die Selektierung einzelner Flurstücke. Sobald das Werkzeug aktiviert ist, können Sie per Mausklick gewünschte Flurstücke aussuchen, welche dann in der Ablage aufgelistet werden.

![](cadastral_unit_searching_5.png)


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

## Protokollierter Zugang zu Personendaten

Zum einbehalten der Datenschutzrechte gibt es folgendes Werkzeug in der Flurstücksuche. Wenn jemand nach personenbezogenen Daten wie Vorname und Name sucht, muss das Häckchen bei  ``Zugang zu Personendaten`` gesetzt werden. Es öffnet sich ein Fenster in dem ein firmeninternes Aktenzeichen vergeben werden muss. Dieses Aktenzeichen verifiziert den Erhalt der personenbezogenen Daten. Außerdem wird jede Anfrag in einer PostGIS Datei abgelegt, sodass jede Anfrage protokolliert und somit kontrolliert werden kann.

![](cadastral_unit_search_data_rights.png)


