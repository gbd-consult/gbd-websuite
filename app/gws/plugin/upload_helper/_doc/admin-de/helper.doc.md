# Helfer "upload" :/admin-de/konfiguration/helper/upload

Der Hilfsdienst `upload` verwaltet stückweise (chunked) Datei-Uploads. Der Client sendet eine Datei in einzelnen Teilstücken; der Dienst setzt sie zwischengespeichert wieder zusammen und stellt die fertige Datei zur weiteren Verarbeitung bereit. Er wird von Aktionen genutzt, die Datei-Uploads entgegennehmen, und bei Bedarf automatisch bereitgestellt. Eintragen müssen Sie ihn nur, um die Höchstgröße eines Uploads zu ändern: `maxSize` gibt sie in Megabyte an, voreingestellt `1000`.

## Beispiel-Konfiguration ::

```javascript
helpers+ { type "upload" maxSize 2000 }
```

%ref "gws.plugin.upload_helper.Config"
