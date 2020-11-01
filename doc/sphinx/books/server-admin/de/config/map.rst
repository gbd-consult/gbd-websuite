Karten
======

^REF gws.common.map.Config

Eine Karte ist im Grunde eine Sammlung von *Layern* (``layers``). Eine Kartenkonfiguration kann auch Ansichtsoptionen enthalten, die als Fallback für Ebenen dienen, die diese nicht explizit definieren. Die Option ``crs`` muss ein gültiger EPSG KBS-Referenzstring sein. Alle Layer im Projekt werden in diesem KBS angezeigt, wobei die Quellen mit unterschiedlichen Projektionen dynamisch umprojiziert werden.

^NOTE In der Zukunft wird es möglich sein, mehrere KBS pro Projekt zu nutzen.

Die Darstellung der Karte wird von der Aktion ``map`` behandelt.

Karten-Ausmaß
-------------

Das Ausmaß der Karte kann mit ``extent`` konfiguriert werden. In Client kann der Nutzer die Karte nicht außerhalb des Extents scrollen. Die Koordinaten des Extents sind in KBS Einheiten anzugeben. Falls Sie kein ``extent`` angeben, wird das Ausmaß aus Layer-Ausmaßen berechnet. Mit ``extentBuffer`` können Sie den automatisch berechneten Extent-Wert erweitern.

Die Anfangsposition der Karte können Sie mit der Option ``center`` konfigurieren, ansonsten wird die Mitte des Extents genommen.

Zoomstufen
----------

Sie können eine Liste von Zoomstufen definieren, um das Zoomen im Client einzugrenzen. Sie können die Stufen entweder als Maßstab (``scales``) Werte (1 Millimeter Bildschirm gleich X Millimeter Erdoberfläche) oder Auflösung (``resolutions``) Werte (1 Pixel Bildschirm gleich X Meter Erdoberfläche) definieren. Bei Maßstab-Angaben beachten Sie bitte, dass es sich um ungefähre Werte handelt, weil die physische Bildschirmgröße dem System grundsätzlich nicht bekannt ist. Bei Pixel-Berechnungen gehen wir von einem von OGC standardisierten Wert aus ``1 pixel = 0.28 x 0.28 mm``.
