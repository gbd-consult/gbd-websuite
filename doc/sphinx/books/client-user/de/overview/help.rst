Funktionsübersicht
==================

Funktionsübersicht der GBD WebSuite.

Die GBD WebSuite ist eine webbasierte Open Source GIS Plattform. Sie beinhaltet den GBD WebSuite Server und GBD WebSuite
Client und zeichnet sich neben der klassischen WebGIS Funktionalität vor allem dadurch aus, externe Anwendungen und
neue Funktionalitäten modular und effizient zu integrieren und umfangreiche Konfigurationen zu ermöglichen. Die
Kern-Bibliotheken von Client und Server sind schlank gehalten. Die gesamte Architektur ist Plugin-basiert. Auch die QGIS
Integration ist als Plugin implementiert, welche nativ mit QGIS kommuniziert.

Die ausführliche Hilfe im Umgang mit der GBD WebSuite erhalten Sie im weiteren Verlauf dieser Dokumentation.

Übersicht der Bedienelemente in der GBD WebSuite:

.. toctree::
    :maxdepth: 2

**Werkzeugleiste**
==================

+------------------------+--------------------------------------------------+-----------------------------------------------------------+
| **Symbol**             | **Funktion**                                     |        **Funktionsbeschreibung**                          |
+------------------------+--------------------------------------------------+-----------------------------------------------------------+
|       |select|         |:ref:`Auswählen <selecting>`                      |Auswählen von Objekten per Mausklick                       |
+------------------------+--------------------------------------------------+-----------------------------------------------------------+
|       |info|           |:ref:`Abfragen <object_identi>`                   |Informationen von Objekten per Mausklick                   |
+------------------------+--------------------------------------------------+-----------------------------------------------------------+
|       |mouseover|      |:ref:`Anzeigen <mouseover>`                       |Informationen von Objekten per Mouseover                   |
+------------------------+--------------------------------------------------+-----------------------------------------------------------+
|  |geo_search|          |:ref:`Räumliche Suche <searching>`                |Suche mit Hilfe von geometrischen Objekten                 |
+------------------------+--------------------------------------------------+-----------------------------------------------------------+
|         |dimensions|   |:ref:`Bemaßung <dimensioning>`                    |Erfassung von Strecken-Distanzen                           |
+------------------------+--------------------------------------------------+-----------------------------------------------------------+
|        |measure|       |:ref:`Markieren & Messen <measure>`               |Markieren mit Hilfe von geometrischen Objekten             |
+------------------------+--------------------------------------------------+-----------------------------------------------------------+
|      |dpro|            | :ref:`Auswahl an D-ProCon übermittlen <dprocon>` |Selektierte Auswahl an Objekten an D-ProCon übermitteln    |
+------------------------+--------------------------------------------------+-----------------------------------------------------------+
|      |gkos|            | :ref:`Auswahl an GeKos übermittlen <gkos>`       |Selektierte Auswahl an Objekten an GeKoS übermitteln       |
+------------------------+--------------------------------------------------+-----------------------------------------------------------+
|      |print|           |    :ref:`Drucken <print>`                        |PDF-Generierung welche gespeichert und gedruckt werden kann|
+------------------------+--------------------------------------------------+-----------------------------------------------------------+
|      |screenshot|      |    :ref:`Screenshot exportieren <screenshot>`    |Abspeichern eines Kartenausschnitts als PNG-Datei          |
+------------------------+--------------------------------------------------+-----------------------------------------------------------+

Mehr über die einzelnen Funktionen finden sie unter: :ref:`Werkzeugleiste <mapfunction>`

**Menüleiste**

+------------------------+------------------------------------------------------+-----------------------------------------+
| **Symbol**             | **Funktion**                                         |          **Funktionsbeschreibung**      |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |menu|            |    Menü ausklappen                                   |Anzeigen des Untermenüs                  |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |project|         |    :ref:`Projektübersicht <project_overview>`        |Übersicht über die Projekteigenschaften  |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |layers|          |    :ref:`Layer <map_element>`                        |Übersicht über die einzelnen Layer       |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |search|          |    :ref:`Suche <search>`                             |Liste von Ergebnissen der Suche          |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |select|          |    :ref:`Auswahl <select>`                           |Liste der ausgewählten Objekte           |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |cadastralunit|   |    :ref:`Flurstückssuche <cadastral_unit_searching>` |Liste der ausgewählten Flurstücke        |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |person_info|     |    :ref:`Personen Informationen <person_info>`       |Personenbezogene Flurstücksinformationen |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |measure|         |   :ref:`Markierungen <markings>`                     |Liste der ausgewählten Markierungen      |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |dimensions|      |   :ref:`Bemaßung <dimensions>`                       |Auflistung der erstellten Bemaßungen     |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |edit|            |   :ref:`Editieren <editing>`                         |Liste der editierbaren Objekte           |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |table_edit|      |   :ref:`Tabelle editieren <table_editing>`           |Öffnet Tabelle zum Attribute editieren   |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |authorization|   |   :ref:`Anmeldung <sign_in>`                         |An- und Abmelden von Benutzern           |
+------------------------+------------------------------------------------------+-----------------------------------------+

Mehr über die einzelnen Funktionen finden sie unter: :ref:`Menüleiste <sidebar>`

**Statusleiste**

+------------------------+--------------------------------------------+-----------------------------------------+
| **Symbol**             | **Funktion**                               |       **Funktionsbeschreibung**         |
+------------------------+--------------------------------------------+-----------------------------------------+
|      |zoomin|          |:ref:`Hineinzoomen <navigation>`            |In die Karte Hineinzoomen                |
+------------------------+--------------------------------------------+-----------------------------------------+
|      |zoomout|         |:ref:`Herauszoomen <navigation>`            |Aus der Karte Herauszoomen               |
+------------------------+--------------------------------------------+-----------------------------------------+
|      |zoommap|         |:ref:`Ganzes Projekt anzeigen <navigation>` |Herauszoomen auf die Ausmaße des Projekts|
+------------------------+--------------------------------------------+-----------------------------------------+
|      |home|            |:ref:`Zurück zur Startseite <home>`         |Öffnet die Anmeldeseite der WebSuite     |
+------------------------+--------------------------------------------+-----------------------------------------+
|      |help|            |:ref:`Hilfe <help>`                         |Öffnet die Dokumentation der GBD WebSuite|
+------------------------+--------------------------------------------+-----------------------------------------+

Mehr über die einzelnen Funktionen finden sie unter: :ref:`Statusleiste <infobar>`

Informationen zur Geoinformatikbüro Dassau GmbH und zur GBD WebSuite finden Sie unter: https://gbd-websuite.de/

   .. |info| image:: ../../../images/gbd-icon-abfrage-01.svg
     :width: 30em
   .. |person_info| image:: ../../../images/person_search_black_24dp.svg
     :width: 30em
   .. |table_edit| image:: ../../../images/table_view_black_24dp.svg
     :width: 30em
   .. |options| image:: ../../../images/round-settings-24px.svg
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
   .. |dpro| image:: ../../../images/gbd-icon-d-procon-02.svg
     :width: 30em
   .. |gkos| image:: ../../../images/gbd-icon-gekos-04.svg
     :width: 30em
   .. |screenshot| image:: ../../../images/outline-insert_photo-24px.svg
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
   .. |help| image:: ../../../images/sharp-help-24px.svg
      :width: 30em
   .. |home| image:: ../../../images/baseline-home-24px.svg
       :width: 30em
   .. |zoomin| image:: ../../../images/zoom-24.svg
         :width: 30em
   .. |zoomout| image:: ../../../images/zoom_out.svg
         :width: 30em
   .. |zoommap| image:: ../../../images/zoom_reset.svg
         :width: 30em
   .. |gbd| image:: ../../../images/gws_logo.svg
        :width: 30em

.. |      |coordinates|     |:ref:`Koordinatenanzeige <coordinates>`     |Zeigt die Koordinaten der Mausposition an|
.. +------------------------+--------------------------------------------+-----------------------------------------+
.. |      |scale|           |:ref:`Maßstab <scale>`                      |Ablesen des aktuellen Kartenmaßstabs     |
.. +------------------------+--------------------------------------------+-----------------------------------------+
.. |      |turn|            |:ref:`Rotation <turn>`                      |Ablesen der aktuellen Kartenrotation     |
.. +------------------------+--------------------------------------------+-----------------------------------------+
