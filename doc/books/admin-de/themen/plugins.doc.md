# Plugins :/admin-de/themen/plugins

Die GBD WebSuite ist modular aufgebaut. Funktionen werden über **Plugins** aktiviert, die in der Konfiguration als `actions` (serverseitige Aktionen) oder `helpers` (Hintergrunddienste) eingebunden werden. Interaktive Plugins benötigen zusätzlich passende Client-Elemente in der `client`-Konfiguration.

## Plugins aktivieren

Die meisten Plugins werden über einen Eintrag in der `actions`-Liste aktiviert. Plugins, die als Hintergrunddienst laufen, werden über die `helpers`-Liste eingebunden:

```javascript
{
    actions+ {
        type "plugin_typ"
    }

    helpers+ {
        type "plugin_typ"
    }
}
```

Damit ein Plugin im Client sichtbar wird, müssen die zugehörigen Client-Elemente freigeschaltet werden:

```javascript
{
    client.addElements+ { tag "Toolbar.PluginName" }
    client.addElements+ { tag "Sidebar.PluginName" }
}
```

Alternativ können Elemente direkt in der `client.elements`-Liste definiert werden, wenn die gesamte Client-Konfiguration neu gesetzt wird (siehe [Client](/admin-de/config/client "Client")).

<div class="admonition_info">Über <code>client.addElements</code> werden Elemente zur bestehenden Client-Konfiguration <em>hinzugefügt</em>. Über <code>client.elements</code> wird die gesamte Liste neu definiert.

</div>

## Annotate

Das Annotate-Plugin erlaubt es Benutzern, Beschriftungen, Texte und einfache Zeichnungen direkt in der Karte zu erstellen und zu speichern.

```javascript
{
    actions+ {
        type "annotate"
    }

    client.addElements+ { tag "Toolbar.Annotate" }
    client.addElements+ { tag "Sidebar.Annotate" }
}
```

Sollen Annotationen dauerhaft gespeichert werden, muss ein Speicher-Provider konfiguriert sein:

```javascript
{
    storage.providers+ {
        type "sqlite"
        path "/gws-var/misc/storage.sqlite"
    }
}
```

## ALKIS

Das ALKIS-Plugin stellt Funktionen zur Abfrage und Darstellung von Daten aus dem Amtlichen Liegenschaftskatasterinformationssystem bereit.

Das Plugin wird über die `helpers`-Liste eingebunden und benötigt Zugriff auf eine PostgreSQL-Datenbank mit ALKIS-Daten sowie einen konfigurierten [Datenbankprovider](/admin-de/themen/postgresql "PostgreSQL"):

```javascript
{
    helpers+ {
        type "alkis"
        dbUid "meine_db"
        crs "EPSG:25832"
    }

    actions+ {
        type "alkis"
    }

    client.addElements+ { tag "Toolbar.Alkis" }
    client.addElements+ { tag "Sidebar.Alkis" }
}
```

## GeKos

Das GeKos-Plugin bietet eine Integration mit dem Fachverfahren GeKos und ermöglicht den Aufruf von GeKos-Vorgängen direkt aus der Karte.

```javascript
{
    actions+ {
        type "gekos"
        url "https://gekos.example.com"
    }

    client.addElements+ { tag "Toolbar.Gekos" }
}
```

## QGIS

Das QGIS-Plugin ermöglicht die Einbindung von QGIS-Projekten als Layer sowie die Verwendung von QGIS-Druckvorlagen.

Eine ausführliche Dokumentation finden Sie unter [QGIS](/admin-de/plugin/qgis "QGIS").

## Auswahl

Das Auswahl-Plugin erlaubt das Markieren und Speichern mehrerer Kartenobjekte durch Klicken oder Polygon-Zeichnen.

Eine ausführliche Dokumentation finden Sie unter [Auswahl](/admin-de/plugin/auswahl "Auswahl").

## Bemaßung

Das Bemaßungs-Plugin ermöglicht das Zeichnen und Speichern von Maßlinien in der Karte, mit optionalem Einrasten an Vektorobjekten.

Eine ausführliche Dokumentation finden Sie unter [Bemaßung](/admin-de/plugin/dimension "Bemaßung").
