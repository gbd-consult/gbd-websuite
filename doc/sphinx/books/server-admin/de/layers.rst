Layer
======

Ein *Layer* in einem GBD WebSuite Projekt wird durch seinen ``type`` identifiziert. Zusätzlich haben die Layer folgenden Eigenschaften (wenn nicht explizit konfiguriert, werden die Eigenschaften von der übergeordneten Ebene oder Map geerbt):

* ``source`` - die Quelle wo die Ebene ihre Geodaten herbekommt (siehe :doc:`sources`)
* ``view`` - räumliche Eigenschaften der Ebene (Umfang, erlaubte Auflösungen oder Skalen für diese Ebene)
* ``cache`` und ``grid`` - beeinflussen das Layer-Caching (siehe :doc:`cache`)
* ``clientOptions`` - Optionen für den GBD WebSuite Client (siehe :doc:`client`)
* ``attribute`` - Layer-Metadaten (z. B. Attribution)
* ``meta`` - Transfomationsregeln für Features (siehe :doc:`features`)

Layer Typen
-------------

Box
~~~

Eine Box-Schicht ist vergleichbar mit einer konventionellen WMS-Schicht. Es wird mit den WMS-Parametern ``bbox``, ``width`` und ``height`` abgefragt und liefert ein ``png`` Bild.

Kachel
~~~~~~

Eine Kachelschicht arbeitet als XYZ-Kachelquelle. Beachten Sie, dass in Abweichung von der allgemeinen Regel, Anfragen an Tile-Layer statische Anfragen imitieren, um clientseitiges Caching zu ermöglichen. Ein Beispiel für die Anforderung einer Kachelschicht ::

    http://example.org/_/cmd/mapHttpGetXyz/layer/project.layer/z/1/x/2/y/3/t.png


Gruppe
~~~~~~~

Gruppenebenen enthalten andere Ebenen, sie liefern selbst keine Geodaten. Neben der visuellen Gruppierung besteht ein weiterer Zweck einer Gruppe darin, die Zugriffs- oder Fallback-Cache- und Grid-Konfigurationen für ihre untergeordneten Ebenen beizubehalten. Eine Gruppe kann "virtuell" oder ``unfolded`` erstellt werden, in diesem Fall wird sie im Client nicht angezeigt, während ihre untergeordneten Ebenen vorhanden sind.


Baum
~~~~

Eine Baumschicht ist in der Lage, eine ganze Hierarchie von Schichten aus einer WMS- oder QGIS-Quelle darzustellen. Eine Baumschicht wird als Gruppe im Client und mit Quellschichten als Unterknoten (oder *leaves*) angezeigt.

Es ist auch möglich, nur bestimmte Ebenen aus der Quelle auszuwählen. Beim Lesen der Quelle erzeugt der Server eine virtuelle *path*-Eigenschaft für jede Schicht, die die eindeutige ID der Schicht und ihre übergeordneten ids enthält, ähnlich den Pfaden des Dateisystems, wie ``/root-layer-id/grandparent-id/parent-id/layer-id``. Das ``pathMatch`` regex kann verwendet werden, um Ebenen mit passenden Pfaden zu filtern.

QGIS
~~~~

QGIS-Schichten sind ähnlich wie Baumschichten, funktionieren aber nur mit QGIS-Karten. Anstelle eines einzelnen ``pathMatch`` können sie eine Liste von Matchregeln haben, die dem Server sagen, wie er mit passenden QGIS-Layern umgehen soll. Sie können z. B. eine bestimmte Ebene "tilify oder einen bestimmten Teilbaum zu einer Ebene "flatten".

Vektor
~~~~~~~

Vektorebenen werden auf dem GBD WebSuite Client gerendert. Wenn eine Vektorebene angefordert wird, sendet der Server die GeoJSON-Liste der Features und Stilbeschreibungen an den Client, der dann das eigentliche Rendering durchführen soll.
