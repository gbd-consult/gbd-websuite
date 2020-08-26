.. _inspire:

Inspire
=======

Das Inspire |bplan| Modul der GBD WebSuite ermöglicht es, einzelnen Kommunen und Gemeinden Bebauungspläne hochzuladen und zu verwalten.
Die hochgeladenen Daten werden dann konform der INSPIRE-Datenstruktur angelegt und können dann in Form von OGC-Diensten, wie zum Beispiel WMS oder WFS Diensten, bereitgestellt werden.
Eine dynamische Übersetzung Ihrer internen Datenstruktur für INSPIRE Konformität ist somit ermöglicht.

In dem Menü :ref:`Layer <layer>` kann man in der Gruppe ``Bauleitplanung`` auswählen aus welcher Kategorie Bebauungspläne angezeigt werden sollen.
Den Kategorien wurden unterschiedliche Farben zugeordnet, mit denen die Geometrien der Bebauungspläne umrandet werden. Außerdem gibt es von jede Kategorie als Vektor- und als Rasterlayer.
Im Rasterlayer befinden sich die eingescannten Bebauungspläne. Im Vektorlayer befinden sich die Umrisse der eingescannten Pläne.
Bei einem Maßstab 1 zu 5000 und größer werden ALKIS Karten als Hintergrundkarten verwendet. Bei einem Maßstab über 1 zu 5000 werden die Karten aus dem Webatlas verwendet.

Wenn man das Modul ``Inspire`` |bplan|  aktiviert, öffnet sich die Liste mit den ausgewählten Bebauungsplänen.
Oberhalb der Liste ist ein Suchfeld angeordnet.
Über dieses Feld ist es möglich Bebauungspläne über ihren Namen zu Suchen. Dabei reicht die Eingabe einzelner Buchstaben des Wortes bereits aus das diese angezeigt werden.
Wenn ein Bebauungsplan angeklickt wird, wird automatisch zu diesem Objekt gezoomt und das dazugehörige Informationsfenster öffnet sich.
In dem Informationsfenster findet man alle angehängten Dokumente, wie zum Beispiel Änderungen oder Artenschutzprüfungen, zu diesem Objekt.
Durch ein Anklicken der Dokumente öffnen sich diese in einem separaten Fenster.
Wenn Sie einen Bebauungsplan löschen wollen, weil zum Beispiel die Eingabe fehlerhaft war, ist dies über das |trash| Icon möglich, welches sich hinter dem jeweiligen Plan befindet.

Unterhalb der Liste befinden sich drei Werkezuge.
Über |new_bplan| kann man neue ``Bauleitpläne hochladen``. Sobald man das Icon anklickt öffnet sich ein neues Dialogfenster.
Hier kann man nun durch Anklicken der Büroklammer, auf die lokalen Daten zugreifen und die gewünschte Datei auswählen.
Die Datei muss eine ZIP-Datei sein und gewissen Vorgaben des übergeordneten Landkreises entsprechen.
Wenn die richtige Datei ausgewählt wurde, kann man auf den Haken drücken und die Datei wird in die Datenbank der GBD WebSuite hochgeladen.
Das Hochladen kann einige Minuten dauern. Falls das Dateiformat oder der Aufbau der ZIP-Datei nicht stimmen, erhält man eine Fehlermeldung.

Das Bearbeiten der Metadaten die andere Kommune und an den Landkreis gesendet werden, ist über |metadata| ``Metadaten editieren`` möglich.
Beim aktivieren dieses Werkzeugs öffnet sich das Formular in welchem die Metadaten eingetragen werden können.
Einige Informationen wie zum Beispiel: wann war die letzte Änderung oder wer hat die letzte Änderung vorgenommen, vermerkt die GBD WebSuite selbstständig.
Andere Informationen wie zum Beispiel Kontaktdaten etc., müssen vom Nutzer selbst eingetragen werden.

Als letztes Werkzeug stehen die |world| ``Dienste`` noch zur Verfügung. Hier kann eine Übersicht über die Links der bereitgestellten Dienste gewonnen werden.
Jede Kommune hat einen eigenen WMS Dienst, welcher im Geoportal integriert werden kann.
Die Zahlen am Ende des Links spiegeln den Geimeindeschlüssel der jeweiligen Kommune wieder.
Wenn am Ende des Links ``Gesamt`` steht, stellt dieser Dienst den gesamten Landkreis da.
So kann der Landkreis die Pläne einer Kommune ganz unkompliziert in den Gesamtbestand einarbeiten und bereitstellen.
Die WMS Dienste können aber auch überall anderes eingebunden werden oder alternativ auch als WMTS oder WCS Dienst angeboten werden.


 .. |bplan| image:: ../../../images/bplan.svg
   :width: 30em
 .. |newline|  image:: ../../../images/baseline-timeline-24px.svg
   :width: 30em
 .. |newpolygon| image:: ../../../images/polygon-create-24px.svg
   :width: 30em
 .. |edit| image:: ../../../images/baseline-create-24px.svg
   :width: 30em
 .. |labelon| image:: ../../../images/baseline-text_format-24px.svg
   :width: 30em
 .. |attribut| image:: ../../../images/baseline-add_box-24px.svg
   :width: 30em
 .. |level| image:: ../../../images/baseline-add-24px.svg
   :width: 30em
 .. |selectedit| image:: ../../../images/baseline-call_made-24px.svg
   :width: 30em
 .. |deleteattributes| image:: ../../../images/baseline-indeterminate_check_box-24px.svg
   :width: 30em
 .. |editstyl| image:: ../../../images/baseline-color_lens-24px.svg
   :width: 30em
 .. |labeloff| image:: ../../../images/text-cancel-24px.svg
   :width: 30em
 .. |menu| image:: ../../../images/baseline-menu-24px.svg
   :width: 30em
 .. |trash| image:: ../../../images/baseline-delete-24px.svg
   :width: 30em
 .. |new_bplan| image:: ../../../images/sharp-control_point-24px.svg
   :width: 30em
 .. |metadata| image:: ../../../images/content_paste-24px.svg
   :width: 30em
 .. |world| image:: ../../../images/language-24px.svg
   :width: 30em
