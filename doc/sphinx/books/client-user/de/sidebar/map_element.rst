Layer
=====


In der Menüebene |layers| ``Layer`` werden die Layer des Projekts angezeigt. Sie können die Sichtbarkeit ein- oder ausschalten. Wenn für bestimmte Layer eine maßstabsbezogene Darstellung konfiguriert ist, erscheinen diese erst wenn der entsprechende Kartenmaßstab erreicht wurde. Um einen Layer zu aktivieren muss auf den jeweiligen Layernamen geklickt werden. Sobald der Layer aktiviert ist, wird der Layername in blau gezeichnet und es öffnet sich die Layerlegende.

.. figure:: ../../../screenshots/de/client-user/layer.png
  :align: center

Layer können in Gruppen zusammengefasst werden. Die Unterebenen werden über den Pfeil links neben dem Gruppennamen geöffnet |showother| und können über |hideother| wieder geschlossen werden.
Das Anzeigen oder Ausblenden eines Layers oder einer Layergruppe erfolgt über einen Klick auf |showlayer| ``Layersichtbarkeit ändern``. Erscheint neben dem Layer das Icon |showlayer| ist dieser sichtbar, erscheint das Icon |hidelayer|, ist er nicht sichtbar. Wenn neben einem Layer das |grouplayer| Icon steht, ist dies ein Zeichen dafür, dass der übergeordnete Gruppenlayer nicht aktiviert ist. Sobald Sie den Gruppenlayer aktivieren, ändert sich das Icon zu |showlayer|.

Mit einem Klick auf den Layernamen öffnet sich die Layerlegende. Hier können die Legende des Layers, die dazugehörigen Metadaten sowie auch Copyright Einträge angezeigt werden. Unter der Layerlegende befinden sich zwei Werkzeuge. Mit |zoom_layer| ``Auf den Layer zoomen`` ist es möglich auf die Gesamtausdehnung des gewählten Layers zu zoomen. Über |cancel| ``Layerlegende schließen`` kann die Layerlegende geschlossen werden.

.. note::
Die Möglichkeiten an Konfigurationen im |layers| ``Layer``-Menü sind sehr umfangreich. Durch die Integration von QGIS in die GBD WebSuite kann z.B. die Layerstruktur aus einem oder mehreren QGIS Projekten mit anderen Datenquellen kombiniert, zusammengefasst und übernommen werden. Im Zusammenspiel mit dem Rechtemanagement ist es zusätzlich möglich, dass verschiedene Nutzer unterschiedliche Layer sehen.



 .. |menu| image:: ../../../images/baseline-menu-24px.svg
   :width: 30em
 .. |showlayer| image:: ../../../images/baseline-visibility-24px-blue.svg
   :width: 30em
 .. |hidelayer| image:: ../../../images/baseline-visibility_off-24px-gray.svg
   :width: 30em
 .. |grouplayer| image:: ../../../images/baseline-visibility-24px-gray.svg
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
