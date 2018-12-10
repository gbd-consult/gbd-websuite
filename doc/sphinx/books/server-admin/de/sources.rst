Quellen
=========

Eine *Quelle* beschreibt, woher die Geodaten stammen. Es gibt verschiedene Arten von Quellen. Beachten Sie, dass nicht alle Quellen mit allen Ebenentypen verwendet werden können. Beispielsweise können Sie keine WMS-Quelle mit einer Vektorebene verwenden.

Quellentypen
------------

Wms
~~~

Eine WMS-Quelle liefert WMS-Bilder von einem externen Dienst. Zusätzlich zu der Service-URL und den Parametern können Sie auch ``maxRequests`` konfigurieren, um zu verhindern, dass die GBD WebSuite den Dienst übermäßig schabt, besonders wenn die Caches gesetzt werden.


Wmts
~~~~

Eine WMTS-Quelle ähnelt dem WMS, arbeitet aber mit WMTS-Diensten.

Kachel
~~~~~~

Kachelquellen arbeiten mit XYZ-Kacheldiensten.

QGIS
~~~~

QGIS-Quellen verwenden QGIS-Maps (``. qgs`` Dateien) als Daten- und Bildquellen. Sie können mehrere QGIS-Quellen in Ihrem Projekt verwenden und diese frei mit anderen Quellentypen kombinieren. Die Option "Ebenen" sagt, welche Ebenen aus ``. qgs`` in das Projekt aufgenommen werden sollen, und setzt ``alle Ebenen`` auf ``wahr``, um alle Ebenen einzuschließen.

Hinweis: QGIS verwaltet Layernamen und ids als separate Felder. Sie können entweder für die ``Layers`` Liste verwenden, aber wenn Sie ids verwenden, stellen Sie sicher, dass die QGIS-Option "Use layer ids as names" aktiviert ist.


geoJSON
~~~~~~~

Eine geoJSON-Quelle ist eine Datei, die mit Vektorebenen verwendet werden kann.

SQL
~~~

Eine SQL-Quelle beschreibt einen DB-Provider und eine Tabelle, aus der die Daten abgerufen werden können. Diese Quelle kann mit Vektorebenen verwendet werden.
