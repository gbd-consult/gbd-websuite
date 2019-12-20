Funktionsübersicht
==================

Die Kurz-Anleitung der GBD WebSuite.

Die GBD WebSuite ist eine webbasierte Open Source GIS Plattform. Sie beinhaltet den GBD WebSuite Server und GBD WebSuite
Client und zeichnet sich neben der klassischen WebGIS Funktionalität vor allem dadurch aus, externe Anwendungen und
neue Funktionalitäten modular und effizient zu integrieren und umfangreiche Konfigurationen zu ermöglichen. Die
Kern-Bibliotheken von Client und Server sind schlank gehalten. Die gesamte Architektur ist Plugin-basiert. Auch die QGIS
Integration ist als Plugin implementiert, welche nativ mit QGIS kommuniziert.

Die ausführliche Hilfe im Umgang mit der GBD WebSuite erhalten Sie im weiteren Verlauf dieser Dokumentation.


Übersicht der Bedienelemente in der GBD WebSuite

Werkzeugleiste:

+------------------------+-----------------------------------------+
| **Symbol**             | **Funktion**                            |
+------------------------+-----------------------------------------+
|       |select|         |:ref:`Auswählen`                         |
+------------------------+-----------------------------------------+
|       |info|           |:ref:`Abfragen`                          |
+------------------------+-----------------------------------------+
|       |mouseover|      |:ref:`Anzeigen`                          |
+------------------------+-----------------------------------------+
|  |geo_search|          | :ref:`Räumliche Suche`                  |
+------------------------+-----------------------------------------+
|         |dimensions|   |                :ref:`Bemaßung`          |
+------------------------+-----------------------------------------+
|         |arrow|        |Bearbeiten                               |
+------------------------+-----------------------------------------+
|            |line|      |    Strecke zeichnen                     |
+------------------------+-----------------------------------------+
|      |trash|           |    Löschen                              |
+------------------------+-----------------------------------------+
|        |measure|       |    :ref:`Markieren & Messen`            |
+------------------------+-----------------------------------------+
|    |point|             |   Punkt markieren                       |
+------------------------+-----------------------------------------+
|    |line|              |   Linie markieren oder messen           |
+------------------------+-----------------------------------------+
|      |quadrat|         |Rechteckige Fläche markieren oder messen |
+------------------------+-----------------------------------------+
|      |polygon|         |Polygone Fläche markieren oder messen    |
+------------------------+-----------------------------------------+
|      |measurecircle|   |Kreisförmige Fläche markieren oder messen|
+------------------------+-----------------------------------------+
|      |trash|           |    Löschen                              |
+------------------------+-----------------------------------------+
|      |continue|        |    Weiter                               |
+------------------------+-----------------------------------------+
|      |back|            |  Zurück                                 |
+------------------------+-----------------------------------------+
|      |fokus|           |    Zu ausgewählten Objekt zoomen        |
+------------------------+-----------------------------------------+
|      |several|         |    Auswahl weiterer Aktionen            |
+------------------------+-----------------------------------------+
|      |dpro|            |    Auswahl an D-ProCon übermittlen      |
+------------------------+-----------------------------------------+
|      |gkos|            |    Auswahl an GeKos übermittlen         |
+------------------------+-----------------------------------------+
|      |print|           |    Drucken                              |
+------------------------+-----------------------------------------+
|      |screenshot|      |    Screenshot exportieren               |
+------------------------+-----------------------------------------+
|      |cancel|          |    Schließen                            |
+------------------------+-----------------------------------------+

Mehr über die einzelnen Funktionen finden sie unter: :ref:`Werkzeugleiste`


Menüleiste

+------------------------+-----------------------------------------+
|      |menu|            |    Menü ausklappen                      |
+------------------------+-----------------------------------------+
|      |project|         |    :ref:`Projektübersicht`              |
+------------------------+-----------------------------------------+
|      |layers|          |    :ref:`Layer`                         |
+------------------------+-----------------------------------------+
|      |search|          |    :ref:`Suche`                         |
+------------------------+-----------------------------------------+
|      |several|         |    Weiteres                             |
+------------------------+-----------------------------------------+
|      |select|          |    :ref:`Auswahl`                       |
+------------------------+-----------------------------------------+
|      |cadastralunit|   |    :ref:`Flurstückssuche`               |
+------------------------+-----------------------------------------+
|      |measure|         |   :ref:`Markierungen`                   |
+------------------------+-----------------------------------------+
|      |dimensions|      |   :ref:`Bemaßung`                       |
+------------------------+-----------------------------------------+
|      |edit|            |    :ref:`Editieren`                     |
+------------------------+-----------------------------------------+
|      |authorization|   |   :ref:`Anmeldung`                      |
+------------------------+-----------------------------------------+

Mehr über die einzelnen Funktionen finden sie unter: :ref:`Menüleiste`


Menüleiste: Projektübersicht

+------------------------+-----------------------------------------+
|      |project|         |    Projektübersicht                     |
+------------------------+-----------------------------------------+


Menüleiste: Layer

+------------------------+-----------------------------------------+
|      |layers|          |    Layer                                |
+------------------------+-----------------------------------------+
|      |showother|       |    Unterlayer aufklappen                |
+------------------------+-----------------------------------------+
|      |hideother|       |    Unterlayer zuklappen                 |
+------------------------+-----------------------------------------+
|      |showlayer|       |    Layer anzeigen                       |
+------------------------+-----------------------------------------+
|      |hidelayer|       |    Layer ausschalten                    |
+------------------------+-----------------------------------------+
|      |zoom_layer|      |   Auf den Layer zoomen                  |
+------------------------+-----------------------------------------+
|      |off_layer|       |   Andere Layer verbergen                |
+------------------------+-----------------------------------------+
|      |cancel|          |    Layerdetails schließen               |
+------------------------+-----------------------------------------+


Menüleiste: Auswahl

+------------------------+-----------------------------------------+
|      |select|          |    Auswahl                              |
+------------------------+-----------------------------------------+
|      |fokus|           |    Zur Auswahl zoomen                   |
+------------------------+-----------------------------------------+
|      |delete|          |    Auswahl aufheben                     |
+------------------------+-----------------------------------------+
|      |save|            |    Auswahl speichern                    |
+------------------------+-----------------------------------------+
|      |load|            |    Auswahl laden                        |
+------------------------+-----------------------------------------+
|      |delete|          |    Auswahl löschen                      |
+------------------------+-----------------------------------------+


Menüleiste: Flurstückssuche

+------------------------+-----------------------------------------+
|      |cadastralunit|   |    Flurstückssuche                      |
+------------------------+-----------------------------------------+
|      |point|           |    Flurstücksauswahl per Klick          |
+------------------------+-----------------------------------------+
|      |line|            |    Flurstücksauswahl per Linie          |
+------------------------+-----------------------------------------+
|      |quadrat|         |    Flurstücksauswahl per Rechteck       |
+------------------------+-----------------------------------------+
|      |measurecircle|   |    Flurstücksauswahl per Kreis          |
+------------------------+-----------------------------------------+
|      |cancel|          |    Flurstücksauswahl abbrechen          |
+------------------------+-----------------------------------------+
|      |search|          |    Suchspalte                           |
+------------------------+-----------------------------------------+
|      |results|         |    Ergebnisspalte                       |
+------------------------+-----------------------------------------+
|      |tab|             |    Ablagespalte                         |
+------------------------+-----------------------------------------+
|      |save|            |    Ablage speichern                     |
+------------------------+-----------------------------------------+
|      |load|            |    Ablage laden                         |
+------------------------+-----------------------------------------+
|      |delete|          |    Ablage leeren                        |
+------------------------+-----------------------------------------+
|      |csv|             |    Ablage als CSV exportieren           |
+------------------------+-----------------------------------------+
|      |print|           |    Ablage drucken                       |
+------------------------+-----------------------------------------+



Menüleiste: Markierungen

+------------------------+-----------------------------------------+
|      |measure|         |    Markierungen                         |
+------------------------+-----------------------------------------+
|      |edit|            |    Bearbeiten                           |
+------------------------+-----------------------------------------+
|      |delete|          |   Markierung Löschen                    |
+------------------------+-----------------------------------------+
|      |trash|           |    Alle löschen                         |
+------------------------+-----------------------------------------+



Menüleiste: Bemaßung

+------------------------+-----------------------------------------+
|      |dimensions|      |    Bemaßung                             |
+------------------------+-----------------------------------------+
|      |save|            |    Speichern                            |
+------------------------+-----------------------------------------+
|      |load|            |    Laden                                |
+------------------------+-----------------------------------------+
|      |delete|          |    Löschen                              |
+------------------------+-----------------------------------------+



Menüleiste: Editieren

+------------------------+-----------------------------------------+
|      |edit|            |    Editieren                            |
+------------------------+-----------------------------------------+
|      |edit|            |    Bearbeiten                           |
+------------------------+-----------------------------------------+
|      |edit|            |    Zeichnen                             |
+------------------------+-----------------------------------------+
|      |cancel|          |    Beenden                              |
+------------------------+-----------------------------------------+




Menüleiste: Anmeldung

+------------------------+-----------------------------------------+
|      |sign_in|         |    Anmeldung                            |
+------------------------+-----------------------------------------+


Menüleiste: Suche

+------------------------+-----------------------------------------+
|      |search|          |    Suche                                |
+------------------------+-----------------------------------------+


Statusleiste

+------------------------+--------------------------------------------------------------------------+
|      |zoom_in|         |    Hineinzoomen                                                          |
+------------------------+--------------------------------------------------------------------------+
|      |zoom_out|        |    Herauszoomen                                                          |
+------------------------+--------------------------------------------------------------------------+
|      |show_project|    |    Ganzes Projekt anzeigen                                               |
+------------------------+--------------------------------------------------------------------------+
|      |start|           |    Zurück zur Startseite                                                 |
+------------------------+--------------------------------------------------------------------------+
|      |help|            |    Hilfe                                                                 |
+------------------------+--------------------------------------------------------------------------+
|      |GBD|             |    Informationen zur Geoinformatikbüro Dassau GmbH und zur GBD WebSuite  |
+------------------------+--------------------------------------------------------------------------+


Mehr über die einzelnen Funktionen finden sie unter: ref`Statusleiste`

Informationen zur Geoinformatikbüro Dassau GmbH und zur GBD WebSuite finden Sie unter: https://gws.gbd-consult.de/

   .. |select| image:: ../../../images/gbd-icon-auswahl-01.svg
     :width: 30em
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
   .. |mouseover| image:: ../../../images/gbd-icon-anzeige-01.svg
     :width: 30em
   .. |geo_search| image:: ../../../images/gbd-icon-raeumliche-suche-01.svg
     :width: 30em
   .. |edit| image:: ../../../images/sharp-edit-24px.svg
     :width: 30em
   .. |navi| image:: ../../../images/Feather-core-move.svg
     :width: 30em
   .. |measure| image:: ../../../images/gbd-icon-markieren-messen-01.svg
     :width: 30em
   .. |dimensions| image:: ../../../images/gbd-icon-bemassung-02.svg
     :width: 30em
   .. |arrow| image:: ../../../images/cursor.svg
     :width: 30em
   .. |line| image:: ../../../images/dim_line.svg
     :width: 30em
   .. |point| image:: ../../../images/g_point.svg
       :width: 30em
   .. |quadrat| image:: ../../../images/g_box.svg
       :width: 30em
   .. |polygon| image:: ../../../images/g_poly.svg
       :width: 30em
   .. |distance| image:: ../../../images/dim_line.svg
       :width: 30em
   .. |cancel| image:: ../../../images/baseline-close-24px.svg
       :width: 30em
   .. |measurecircle| image:: ../../../images/dim_circle.svg
       :width: 30em
   .. |trash| image:: ../../../images/baseline-delete-24px.svg
       :width: 30em
   .. |continue| image:: ../../../images/baseline-chevron_right-24px.svg
     :width: 30em
   .. |back| image:: ../../../images/baseline-keyboard_arrow_left-24px.svg
     :width: 30em
   .. |fokus| image:: ../../../images/sharp-center_focus_weak-24px.svg
     :width: 30em
   .. |dpro| image:: ../../../images/gbd-icon-d-procon-02.svg
     :width: 30em
   .. |gkos| image:: ../../../images/gbd-icon-gekos-04.svg
     :width: 30em
   .. |print| image:: ../../../images/baseline-print-24px.svg
     :width: 30em
   .. |screenshot| image:: ../../../images/outline-insert_photo-24px.svg
     :width: 30em
   .. |menu| image:: ../../../images/baseline-menu-24px.svg
       :width: 30em
   .. |search| image:: ../../../images/baseline-search-24px.svg
      :width: 30em
   .. |project| image:: ../../../images/map-24px.svg
     :width: 30em
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
   .. |menu| image:: ../../../images/baseline-menu-24px.svg
         :width: 30em
   .. |showlayer| image:: ../../../images/baseline-visibility-24px.svg
         :width: 30em
   .. |hidelayer| image:: ../../../images/baseline-visibility_off-24px.svg
         :width: 30em
   .. |layers| image:: ../../../images/baseline-layers-24px.svg
         :width: 30em
   .. |showother| image:: ../../../images/baseline-chevron_right-24px.svg
         :width: 30em
   .. |hideother| image:: ../../../images/baseline-expand_more-24px.svg
         :width: 30em
   .. |cancel| image:: ../../../images/baseline-close-24px.svg
         :width: 30em
   .. |zoom_layer| image:: ../../../images/baseline-zoom_out_map-24px.svg
         :width: 30em
   .. |off_layer| image:: ../../../images/sharp-layers_clear-24px.svg
         :width: 30em
   .. |edit_layer| image:: ../../../images/baseline-create-24px.svg
         :width: 30em
   .. |several| image:: ../../../images/more_horiz-24px.svg
         :width: 30em
   .. |authorization| image:: ../../../images/baseline-person-24px.svg
       :width: 30em
