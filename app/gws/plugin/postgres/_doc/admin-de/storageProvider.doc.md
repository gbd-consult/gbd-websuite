# Speicher-Provider "postgres" :/admin-de/konfiguration/storageProvider/postgres

Dieser Storage-Provider legt die Daten der Datenablage in einer PostgreSQL-Tabelle ab. Damit teilen sich mehrere WebSuite-Instanzen denselben Ablage-Bestand.

Mit `dbUid` verweisen Sie auf den zu nutzenden Datenbank-Provider; mit `tableName` bestimmen Sie die Tabelle, in der die Einträge gespeichert werden. Die Tabelle muss bereits mit dem passenden Schema vorhanden sein.

## Beispiel-Konfiguration ::

```javascript
storage.providers+ {
    type "postgres"
    dbUid "main_db"
    tableName "public.gws_storage"
}
```

Der Provider wird über `storage.providers+` aktiviert. `dbUid` verweist auf den zu nutzenden Datenbank-Provider; ist nur ein Datenbank-Provider konfiguriert, kann `dbUid` entfallen. `tableName` benennt die Tabelle für die Ablage-Einträge. Die Tabelle muss mit dem passenden Schema bereits vorhanden sein.

%ref "gws.plugin.postgres.storage_provider.Config"
