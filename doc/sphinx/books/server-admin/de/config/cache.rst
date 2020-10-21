Caching-Framework
=================

Der GWS Server kann Geo-Bilder aus externen Quellen auf der Festplatte zwischenspeichern (or *cachen*), sodass weitere Anfragen zu der selben Quelle viel schneller ausgeführt werden können.  Das Cache Verzeichnis befindet sich in dem von Ihnen konfigurierten *var* Verzeichnis und kann bei Bedarf jederzeit komplett gelöscht werden

^NOTE Die Caches können sehr groß werden und ganz viele Dateien enthalten. Sorgen Sie dafür, dass Ihr Dateisystem ausreichen freien Platz und freie **inodes** hat.

Layer Konfiguration
-------------------

Das Cachen kann für jeden Layer mit den Optionen ``cache`` und ``grid`` flexibel konfiguriert werden.

cache
~~~~~

^REF gws.common.layer.types.CacheConfig

grid
~~~~

^REF gws.common.layer.types.GridConfig

Seeding
-------

^REF gws.common.application.SeedingConfig
^CLIREF cache.seed

Sobald der Cache eingerichtet ist, wird er automatisch gefüllt, wenn Benutzer Ihre Karten in Browser anschauen. Sie können den Cache auch mit den Kommandozeilen-Tools ``gws cache`` befüllen (sogenanntes *Seeding*).

Verwaltung von Cache
--------------------

^CLIREF cache.clean

Mit dem selben Tool können Sie den Status des Cache abfragen oder individuelle Caches löschen.

^NOTE Wenn Sie Ansichts- oder Rasterkonfigurationen ändern, müssen Sie den Cache für die Ebene oder die Karte entfernen, um unangenehme Artefakte zu vermeiden.
