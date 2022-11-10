# Funktionsübersicht :/user-de/help

Funktionsübersicht der GBD WebSuite.

Die GBD WebSuite ist eine webbasierte Open Source GIS Plattform. Sie beinhaltet den GBD WebSuite Server und GBD WebSuite
Client und zeichnet sich neben der klassischen WebGIS Funktionalität vor allem dadurch aus, externe Anwendungen und
neue Funktionalitäten modular und effizient zu integrieren und umfangreiche Konfigurationen zu ermöglichen. Die
Kern-Bibliotheken von Client und Server sind schlank gehalten. Die gesamte Architektur ist Plugin-basiert. Auch die QGIS
Integration ist als Plugin implementiert, welche nativ mit QGIS kommuniziert.

Die ausführliche Hilfe im Umgang mit der GBD WebSuite erhalten Sie im weiteren Verlauf dieser Dokumentation.

Übersicht der Bedienelemente in der GBD WebSuite

## Werkzeugleiste

| Symbol                                | Funktion                   | Funktionsbeschreibung                                       |
|---------------------------------------|----------------------------|-------------------------------------------------------------|
| ![](gbd-icon-auswahl-01.svg)          | [](/user-de/selecting)     | Auswählen von Objekten per Mausklick                        |
| ![](gbd-icon-abfrage-01.svg)          | [](/user-de/object_identi) | Informationen von Objekten per Mausklick                    |
| ![](gbd-icon-anzeige-01.svg)          | [](/user-de/mouseover)     | Informationen von Objekten per Mouseover                    |
| ![](gbd-icon-raeumliche-suche-01.svg) | [](/user-de/search)        | Suche mit Hilfe von geometrischen Objekten                  |
| ![](gbd-icon-bemassung-02.svg)        | [](/user-de/dimensioning)  | Erfassung von Strecken-Distanzen                            |
| ![](gbd-icon-markieren-messen-01.svg) | [](/user-de/measure)       | Markieren mit Hilfe von geometrischen Objekten              |
| ![](gbd-icon-d-procon-02.svg)         | [](/user-de/dprocon)       | Selektierte Auswahl an Objekten an D-ProCon übermitteln     |
| ![](gbd-icon-gekos-04.svg)            | [](/user-de/gekos)         | Selektierte Auswahl an Objekten an GeKoS übermitteln        |
| ![](baseline-print-24px.svg)          | [](/user-de/print)         | PDF-Generierung welche gespeichert und gedruckt werden kann |
| ![](outline-insert_photo-24px.svg)    | [](/user-de/screenshot)    | Abspeichern eines Kartenausschnitts als PNG-Datei           |















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
|      |cadastralunit|   |    :ref:`Flurstücksuche <cadastral_unit_searching>` |Liste der ausgewählten Flurstücke        |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |measure|         |   :ref:`Markierungen <markings>`                     |Liste der ausgewählten Markierungen      |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |dimensions|      |   :ref:`Bemaßung <dimensions>`                       |Auflistung der erstellten Bemaßungen     |
+------------------------+------------------------------------------------------+-----------------------------------------+
|      |edit|            |   :ref:`Editieren <editing>`                         |Liste der editierbaren Objekte           |
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

.. |addall| image:: ../../../images/gbd-icon-alle-ablage-01.svg
.. |add| image:: ../../../images/sharp-control_point-24px.svg
.. |arrow| image:: ../../../images/cursor.svg
.. |authorization| image:: ../../../images/baseline-person-24px.svg
.. |back| image:: ../../../images/baseline-keyboard_arrow_left-24px.svg
.. |cadastralunit| image:: ../../../images/gbd-icon-flurstuecksuche-01.svg
.. |cancel| image:: ../../../images/baseline-close-24px.svg
.. |continue| image:: ../../../images/baseline-chevron_right-24px.svg
.. |csv| image:: ../../../images/sharp-grid_on-24px.svg
.. |delete_shelf| image:: ../../../images/sharp-delete_forever-24px.svg
.. |delete| image:: ../../../images/sharp-remove_circle_outline-24px.svg
.. |distance| image:: ../../../images/dim_line.svg
.. |edit_layer| image:: ../../../images/baseline-create-24px.svg
.. |edit| image:: ../../../images/sharp-edit-24px.svg
.. |fokus| image:: ../../../images/sharp-center_focus_weak-24px.svg
.. |gbd| image:: ../../../images/gws_logo.svg
.. |help| image:: ../../../images/sharp-help-24px.svg
.. |hidelayer| image:: ../../../images/baseline-visibility_off-24px.svg
.. |hideother| image:: ../../../images/baseline-expand_more-24px.svg
.. |home| image:: ../../../images/baseline-home-24px.svg
.. |layers| image:: ../../../images/baseline-layers-24px.svg
.. |line| image:: ../../../images/dim_line.svg
.. |load| image:: ../../../images/gbd-icon-ablage-oeffnen-01.svg
.. |measurecircle| image:: ../../../images/dim_circle.svg
.. |menu| image:: ../../../images/baseline-menu-24px.svg
.. |navi| image:: ../../../images/Feather-core-move.svg
.. |new_search|  image:: ../../../images/baseline-delete_sweep-24px.svg
.. |off_layer| image:: ../../../images/sharp-layers_clear-24px.svg
.. |options| image:: ../../../images/round-settings-24px.svg
.. |point| image:: ../../../images/g_point.svg
.. |polygon| image:: ../../../images/g_poly.svg
.. |project| image:: ../../../images/map-24px.svg
.. |quadrat| image:: ../../../images/g_box.svg
.. |results| image:: ../../../images/baseline-menu-24px.svg
.. |save| image:: ../../../images/sharp-save-24px.svg
.. |search| image:: ../../../images/baseline-search-24px.svg
.. |several| image:: ../../../images/more_horiz-24px.svg
.. |showlayer| image:: ../../../images/baseline-visibility-24px.svg
.. |showother| image:: ../../../images/baseline-chevron_right-24px.svg
.. |spatial_search| image:: ../../../images/gbd-icon-raeumliche-suche-01.svg
 .. |tab| image:: ../../../images/sharp-bookmark_border-24px.svg
.. |trash| image:: ../../../images/baseline-delete-24px.svg
 .. |zoom_layer| image:: ../../../images/baseline-zoom_out_map-24px.svg
.. |zoomin| image:: ../../../images/zoom-24.svg
 .. |zoommap| image:: ../../../images/zoom_reset.svg
.. |zoomout| image:: ../../../images/zoom_out.svg

.. |      |coordinates|     |:ref:`Koordinatenanzeige <coordinates>`     |Zeigt die Koordinaten der Mausposition an|
.. +------------------------+--------------------------------------------+-----------------------------------------+
.. |      |scale|           |:ref:`Maßstab <scale>`                      |Ablesen des aktuellen Kartenmaßstabs     |
.. +------------------------+--------------------------------------------+-----------------------------------------+
.. |      |turn|            |:ref:`Rotation <turn>`                      |Ablesen der aktuellen Kartenrotation     |
.. +------------------------+--------------------------------------------+-----------------------------------------+
