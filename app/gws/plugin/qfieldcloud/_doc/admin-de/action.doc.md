# Aktion "qfieldcloud" :/admin-de/konfiguration/action/qfieldcloud

Die Aktion `qfieldcloud` stellt die Schnittstelle zur QFieldCloud bereit und ermöglicht den Abgleich von Projekten zwischen der GBD WebSuite und der QField-App für die mobile Felderfassung. Wie Sie das zugehörige QGIS-Projekt vorbereiten und wie der Abgleich abläuft, beschreibt das Thema [](/admin-de/themen/fachmodule/qfieldcloud).

Die Aktion kennt nur eine Eigenschaft: Unter `projects` listen Sie die Projekte auf, die in der App zur Auswahl stehen sollen. Die Aktion lässt sich in einem Projekt oder global in der Applikation konfigurieren.

## Ein Projekteintrag

Jeder Eintrag unter `projects` beschreibt ein QField-Projekt:

| Option | Bedeutung |
|---|---|
| `uid` | Kennung des Projekts. Unter dieser Kennung erscheint es in der Schnittstelle und als Verzeichnisname im Arbeitsbereich. |
| `title` | Name, den die App in der Projektliste anzeigt. Ohne Angabe wird die `uid` verwendet. |
| `provider` | Das QGIS-Projekt, das abgeglichen wird. |
| `models` | Datenmodelle für die editierbaren Tabellen. Optional. |
| `mapCacheLifeTime` | Gültigkeitsdauer des Kartencaches, Vorgabe `0` (kein Cache). |
| `access` | Zugriffsregeln – wer dieses Projekt in der App sieht und abgleichen darf. |

Das QGIS-Projekt geben Sie im `provider` an, entweder als Datei über `path` oder aus einer Datenbank über `dbUid`, `schema` und `projectName` – dieselben Angaben wie beim Layer-Typ [`qgis`](/admin-de/konfiguration/layer/qgis). Für Anhänge und mitgelieferte Verzeichnisse ist die Dateivariante nötig, weil sich relative Pfade nur gegen eine Projektdatei auflösen lassen.

## Datenmodelle

`models` ist optional. Findet die WebSuite zu einer editierbaren Tabelle des QGIS-Projekts kein passendes Modell, erzeugt sie selbst eines: vom Typ `postgres`, mit allen Spalten der Tabelle und uneingeschränkt bearbeitbar. Für einen ersten Versuch genügt daher der `provider`.

Ein eigenes Modell konfigurieren Sie, sobald Sie mehr Kontrolle brauchen:

```javascript
models+ {
    type "postgres"
    tableName "erfassung.baum"
    withAutoFields true
    isEditable true
    permissions.edit "allow feldkraft, deny all"
}
```

Die Zuordnung erfolgt über `tableName`; der Name muss auf dieselbe Tabelle zeigen wie der Layer im QGIS-Projekt. `withAutoFields` übernimmt die übrigen Spalten, ohne dass Sie jedes Feld einzeln aufführen. `isEditable` muss gesetzt sein, sonst nimmt die WebSuite den Layer aus dem Paket – es sei denn, er ist in QFieldSync vollständig gesperrt und damit ohnehin nur zum Lesen bestimmt. Mit `permissions.edit` schränken Sie das Schreiben auf Rollen ein.

## Dateianhänge

Fotos und andere Anhänge, die in QField an ein Objekt gehängt werden, landen in der Datenbank, wenn das Modell ein Feld vom Typ `file` enthält. `nameColumn` nimmt den Dateinamen auf, `contentColumn` den Inhalt:

```javascript
models+ {
    type "postgres"
    tableName "edit.district_photo"
    isEditable true
    fields+ {
        type "file"
        name "image_file"
        contentColumn "image_content"
        nameColumn "image"
    }
}
```

QField überträgt einen Anhang in zwei Schritten: Zuerst kommen die Änderungen am Objekt selbst, in denen der Dateiname steht, danach der Inhalt in einer eigenen Anfrage. Die WebSuite findet das zugehörige Objekt, indem sie den Namen aus der zweiten Anfrage mit `nameColumn` vergleicht, und schreibt die Daten in `contentColumn`. Fehlt `nameColumn`, lässt sich der Anhang nicht zuordnen und geht verloren.

## Zugang für die QField-App

Die Aktion beantwortet die Anfragen der App unter dem internen Befehl `qfieldcloudApi`. Damit die App eine gewöhnliche Adresse ansprechen kann, legen Sie in der Webseite eine Rewrite-Regel an, die einen Pfad Ihrer Wahl auf diesen Befehl abbildet und dabei das Projekt benennt:

```javascript
web.sites+ {
    rewriteRules+ {
        pattern "^/qfc/(.*)"
        target "/_/qfieldcloudApi/projectUid/qfield_demo/$1"
    }
}
```

`projectUid` benennt das **Projekt der WebSuite**, nicht das QField-Projekt aus `projects`. Steht die Aktion in einem Projekt, darf die Angabe entfallen; geben Sie sie an, muss sie zu diesem Projekt passen. Ist die Aktion global in der Applikation konfiguriert, ist sie zwingend – sonst weiß die Schnittstelle nicht, in welchem Projektkontext sie arbeitet.

In der App wählen die Nutzer *QFieldCloud-Projekte*, tragen als Server die daraus entstehende Adresse ein – hier `https://example.com/qfc` – und melden sich mit ihren WebSuite-Zugangsdaten an. Anschließend laden sie das Projekt aus der Liste herunter und übertragen ihre Änderungen später über *Synchronisieren* zurück. Die Anmeldung müssen Sie nicht konfigurieren: Die Aktion bringt eine eigene Authentifizierungsmethode mit und registriert sie selbst. Vorhanden sein müssen nur ein [](/admin-de/konfiguration/authProvider), gegen den die Zugangsdaten geprüft werden, und eine [](/admin-de/konfiguration/authSessionManager), deren Lebensdauer bestimmt, wie lange eine Anmeldung der App gültig bleibt.

## Kartencache

Die Hintergrundkacheln eines Projekts rendert der QGIS-Server und die WebSuite legt sie unter `<GWS_VAR_DIR>/qfieldcloud/projects/<uid>/cache` ab – `<uid>` ist dabei die Kennung des QField-Projekts; die fertigen Pakete liegen als `package_*` daneben. `mapCacheLifeTime` bestimmt, wie lange die Kacheln als gültig gelten – voreingestellt ist `0`, also kein Cache. Da das Rendern der Kacheln der aufwendigste Teil der Paketerstellung ist, lohnt hier ein Wert wie `7d`.

## Kleinstmögliche Konfiguration ::

Mehr als das QGIS-Projekt ist nicht nötig; Modelle und Berechtigungen ergänzen Sie später:

```javascript
actions+ {
    type "qfieldcloud"
    projects+ {
        uid "erfassung"
        provider.path "/data/qgis/baeume.qgs"
    }
}
```

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "qfieldcloud"
    access "allow all"
    projects+ {
        uid "qfield_demo"
        title "QField Demo"
        provider.path "/data/qgis/poi.qgs"
        mapCacheLifeTime "7d"
        access "allow all"
        models+ {
            type "postgres"
            tableName "edit.poi"
            withAutoFields true
            isEditable true
            permissions.edit "allow user, deny all"
        }
    }
}
```

Unter `projects` binden Sie je Eintrag ein QGIS-Projekt (`provider.path`) an, das mit QField abgeglichen wird. `models` beschreibt die editierbaren Tabellen des Projekts; `isEditable` und `permissions.edit` steuern, wer sie bearbeiten darf. `mapCacheLifeTime` legt fest, wie lange der Kartencache des Projekts gültig bleibt.

%ref "gws.plugin.qfieldcloud.action.Config"
%demo "qfield_demo"
