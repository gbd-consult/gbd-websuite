QGIS Support
============

Die GBD WebSuite unterstützt und supportet `QGIS <https://qgis.org>`_, ein kostenloses und quelloffenes geografisches Informationssystem. Die Unterstützung ist optional und kann abgeschaltet werden, wenn Sie QGIS nicht verwenden.

Mit unserem QGIS-Modul können Sie

- komplette QGIS-Karten (``. qgs``) als *Layer* in Ihren Projekten anzeigen (siehe :doc:`Layer`)
- QGIS-Karten (``. qgs``) als *Quellen* für Ihre Bildebenen verwenden (siehe :doc:`sources`)
- QGIS-Templates zum Drucken verwenden (siehe :doc:`print`)

In der Serverkonfiguration (:doc:`server`) gibt es einige Optionen, die die Werte von QGIS-Umgebungsvariablen setzen. Die genaue Bedeutung entnehmen Sie bitte der QGIS-Dokumentation:

TABLE
*debug*	~ QGIS_DEBUG
*maxCacheLayers* ~ MAX_CACHE_LAYERS
*serverCacheSize* ~ QGIS_SERVER_CACHE_SIZE
*serverLogLevel* ~ QGIS_SERVER_LOG_LEVEL
/TABLE


Die Option ``searchPathsForSVG`` sagt Ihnen, wo Sie die verwendeten SVG-Bilder in Ihren QGIS-Karten und Druckvorlagen finden. Wenn Sie nicht standardmäßige Bilder verwenden, fügen Sie einfach einen Verzeichnispfad, für die von Ihnen verwendeten SVG-Bilder, zu dieser Einstellung hinzu.
