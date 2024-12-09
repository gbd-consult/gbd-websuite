# Auswahl:/admin-de/plugin/auswahl

## Konfigurationsmöglichkeiten

Mit dem Auswahlwerkzeug kann eine Auswahl mehrerer Objekte markiert und gespeichert werden. Die Auswahl kann per Direktanwahl eines Objektes oder über ein gezeichnetes Polygon stattfinden.

Sie funktioniert mit jeder Art von Layer, solange dieser bei der Abfrage geometrische Informationen zurücksenden kann. 
Um das sicher zu stellen sollte für die entsprechenden Layer `provider.directSearch`  konfiguriert sein:

```javascript title="/data/client.cx"
map.layers+ {
    title "Beispiellayer"
    type "qgis"
    provider.path "/data/projects/beispiel/beispiel.qgs"
    provider.directSearch ["wms" "postgres"]
}
```

Das Zuschalten der Client-Elemente ist zum Nutzen des Tools notwendig.

```javascript title="/data/client.cx"
client.addElements+ { tag "Sidebar.Select" }
client.addElements+ { tag "Toolbar.Select" }
client.addElements+ { tag "Toolbar.Select.Draw" }
client.addElements+ { tag "Task.Select" }
```

Um eine Auswahl abspeichern zu können, muss ein Speicherprovider definiert werden, z.B.:

```javascript title="/data/helpers.cx"
storage.providers+ {
    type "sqlite"
    path "/gws-var/misc/storage8.sqlite"
}
```

Die Speicherberechtigungen können in der Konfiguration der Auswahlaktion definiert werden:

```javascript title="/data/projects/projectname.cx"
actions+ {
    type "select"
    storage {
        permissions {
            read "allow all"
            write "allow all"
            create "allow all"
        }
    }
}
```

