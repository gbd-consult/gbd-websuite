# Speicher-Provider "sqlite" :/admin-de/konfiguration/storageProvider/sqlite

Dieser Storage-Provider legt die Daten der Datenablage in einer lokalen SQLite-Datei ab. Das ist die einfachste Variante und benötigt keine externe Datenbank.

Mit `path` bestimmen Sie den Speicherort der Datei. Ohne Angabe wird eine Standarddatei im internen Verzeichnis der WebSuite verwendet; die benötigte Tabelle wird bei Bedarf automatisch angelegt.

## Beispiel-Konfiguration ::

```javascript
storage.providers+ {
    type "sqlite"
    path "/data/storage.sqlite"
}
```

Der Provider wird über `storage.providers+` aktiviert. `path` bestimmt den Speicherort der SQLite-Datei; ohne Angabe wird eine Standarddatei im internen Verzeichnis der WebSuite verwendet. Die benötigte Tabelle wird beim ersten Zugriff automatisch angelegt.

%ref "gws.plugin.storage_provider.sqlite.Config"
