QGIS Support
============

Die GBD WebSuite bietet dedizierten Support für `QGIS <https://qgis.org>`_, ein kostenloses und quelloffenes geografisches Informationssystem. Die Unterstützung ist optional und kann abgeschaltet werden, wenn Sie QGIS nicht verwenden.

Mit unserem QGIS-Modul können Sie

- komplette QGIS Projekte (``. qgs``) als *Layer* in Ihren GBD WebSuite Projekten anzeigen (siehe :doc:`layers`)
- bestimmte QGIS Layer aus einem QGIS Projekt in Ihre GBD WebSuite Projekte einbinden  
- QGIS-Templates zum Drucken verwenden (siehe :doc:`print`)

In der Serverkonfiguration (:doc:`server`) gibt es einige Optionen, die die Werte von QGIS-Umgebungsvariablen setzen. Die genaue Bedeutung entnehmen Sie bitte der QGIS-Dokumentation:

TABLE
*debug*	~ QGIS_DEBUG
*maxCacheLayers* ~ MAX_CACHE_LAYERS
*serverCacheSize* ~ QGIS_SERVER_CACHE_SIZE
*serverLogLevel* ~ QGIS_SERVER_LOG_LEVEL
/TABLE


Die Option ``searchPathsForSVG`` sagt Ihnen, wo Sie svg-Bilder in Ihren QGIS-Karten und Druckvorlagen finden. Wenn Sie nicht standardmäßige Bilder verwenden, fügen Sie einfach einen Verzeichnispfad für sie zu dieser Einstellung hinzu.

