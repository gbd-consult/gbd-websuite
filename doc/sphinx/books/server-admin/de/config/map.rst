Karten
======

^REF gws.common.map.Config

Eine *Karte* ist im Grunde eine Sammlung von *Ebenen* (``layers``). Eine Kartenkonfiguration kann auch Cache-, Raster- und Ansichtsoptionen enthalten, die als Fallback für Ebenen dienen, die diese nicht explizit definieren. Die Option ``crs`` muss ein gültiger EPSG KBS-Referenzstring sein. Alle Ebenen im Projekt werden in diesem KBS angezeigt, wobei die Quellen mit unterschiedlichen Projektionen dynamisch umprojiziert werden.

^NOTE In der Zukunft wird es möglich sein, mehrere KBS pro Projekt zu haben.

Die Anfangsposition der Karte kann mit der Option ``center`` konfiguriert werden.

Karten-Ausmaß
-------------

Zoomstufen und Auflösungen
--------------------------

Aktion ``map``
--------------

^REF gws.ext.action.map.Config
