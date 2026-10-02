# Client :/admin-de/konfiguration/client

Der Client ist die JavaScript-Anwendung, die die Karte im Browser darstellt. Welche Bausteine seine Oberfläche zeigt, bestimmt eine geordnete Liste von *Elementen*. Jedes Element wird über ein `tag` benannt; die Reihenfolge in der Liste ist die Reihenfolge in der Oberfläche.

Ein Client-Objekt gibt es global in der Applikation und je Projekt. Näheres zum Aufbau der Oberfläche unter [](/admin-de/themen/darstellung/client).

## Client in eine Seite einbinden

Der Client läuft in einer gewöhnlichen HTML-Seite. Diese liefert die Vorlage mit dem Subject `project.home`, die unter `/project/<Projekt-uid>` erreichbar ist; ohne eigene Vorlage verwendet die WebSuite die mitgelieferte. Eine eigene Seite muss drei Dinge enthalten:

```html
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"/>

    <link rel="stylesheet" href="/_/webSystemAsset/path/light.css"/>
    <script src="/_/webSystemAsset/path/vendor.js"></script>
    <script src="/_/webSystemAsset/localeUid/de_DE/path/app.js"></script>

    <script id="gwsOptions" type="application/json">
        { "projectUid": "meinprojekt" }
    </script>
</head>

<body>
    <div class="gws"></div>
</body>
```

Die Dateien des Clients werden unter `/_/webSystemAsset` ausgeliefert; diese Adresse müssen Sie nicht eigens einrichten. `vendor.js` enthält die Bibliotheken, `app.js` die Anwendung, `light.css` das Aussehen. Die Sprache der Oberfläche bestimmt der Pfadteil `localeUid` vor `app.js` – die Texte stecken im Bundle, nicht in der Konfiguration. Den Dateinamen können Sie die Versionsnummer voranstellen (`8.4.5.vendor.js`), um Browser-Caches beim Versionswechsel zu umgehen.

Das Element mit der Klasse `gws` nimmt die Oberfläche auf. Sie können es frei positionieren; fehlt es, hängt der Client selbst eines an das Ende von `<body>`.

Das Skript mit der id `gwsOptions` übergibt die Startwerte als JSON:

| Option | Bedeutung |
|---|---|
| `projectUid` | uid des Projekts, das geladen wird – die einzige zwingende Angabe |
| `serverUrl` | Adresse des Server-Endpunkts, voreingestellt `/_`; nötig, wenn die Seite von einem anderen Host ausgeliefert wird als die WebSuite |
| `showLayers` | Liste von Layer-Kennungen, die beim Start eingeschaltet werden |
| `hideLayers` | Liste von Layer-Kennungen, die beim Start ausgeschaltet werden |
| `markFeatures` | Objekte, die beim Start hervorgehoben und angesteuert werden |
| `customStrings` | überschreibt einzelne Texte der Oberfläche |
| `helpUrl` | Ziel der Hilfe-Schaltfläche; ohne Angabe das Benutzerhandbuch zur laufenden Version |
| `helpUrlTarget` | `blank` (neues Fenster) oder `frame`, voreingestellt `blank` |
| `homeUrl` | Ziel des Verweises auf die Startseite, voreingestellt `/` |

Layer sprechen Sie in `showLayers` und `hideLayers` mit ihrer vollständigen Kennung an, also `<Projekt>.map.<Layer>`. Sind unter `markFeatures` Objekte angegeben, verdrängen sie die Startparameter `x`/`y`/`z`, `bbox` und `extents` aus der Adresszeile. `helpUrl`, `helpUrlTarget` und `homeUrl` lassen sich auch in den `options` des Client-Objekts setzen; die Konfiguration hat dann Vorrang vor der Seite.

%info
Wollen Sie lediglich eigenes CSS oder JavaScript ergänzen, brauchen Sie keine eigene Vorlage: Die Applikation nimmt über `templateOptions.projectResources` zusätzliche Dateien in die Standard-Projektseite auf.
%end

## Oberfläche voreinstellen

Neben den Elementen nimmt das Client-Objekt `options` entgegen. Damit setzen Sie den Zustand, in dem die Oberfläche startet:

| Option | Bedeutung |
|---|---|
| `sidebarVisible` | ob die Seitenleiste beim Start geöffnet ist |
| `sidebarActiveTab` | Tag des Eintrags, der beim Start ausgewählt ist, etwa `Sidebar.Layers` |
| `sidebarWidth` | Breite der Seitenleiste in Pixeln, voreingestellt `300` |
| `toolbarSize` | Anzahl der Schaltflächen, die in der Werkzeugleiste Platz finden |
| `toolbarActiveButton` | Tag der Schaltfläche, deren Werkzeug beim Start aktiv ist |

```javascript
client.options {
    sidebarVisible true
    sidebarActiveTab "Sidebar.Layers"
    sidebarWidth 380
    toolbarSize 8
}
```

Auch `helpUrl`, `helpUrlTarget` und `homeUrl` lassen sich hier setzen; sie gehen dann den Angaben der Seite vor.

%warn
`toolbarSize` verbirgt Schaltflächen, es entfernt sie nicht. Was über der angegebenen Zahl liegt, wandert in ein Überlaufmenü am Ende der Leiste. Vermissen Sie ein Werkzeug, das in `elements` steht, zählen Sie zuerst die `Toolbar`-Einträge vor ihm — bei `toolbarSize 5` ist alles ab dem sechsten nur noch über das Überlaufmenü erreichbar.
%end

%info
`options` ist ein freies Objekt und wird **nicht** geprüft. Ein Tippfehler und eine Option, die es nicht mehr gibt, verhalten sich gleich: Sie werden stillschweigend übernommen und bleiben wirkungslos. Beim Übernehmen einer Konfiguration aus einer älteren Version lohnt daher ein Abgleich mit der Tabelle oben.
%end

## Elemente festlegen

Ein Projekt erbt die Elementliste der Applikation. Drei Eigenschaften steuern, was daraus wird:

| Eigenschaft | Wirkung |
|---|---|
| `elements` | **ersetzt** die Liste vollständig |
| `addElements` | **ergänzt** die geerbte Liste |
| `removeElements` | entfernt einzelne Einträge aus der geerbten Liste |

`addElements` und `removeElements` wirken nur im **Projekt**, gegen die von der Applikation geerbte Liste. Die Liste der Applikation selbst legen Sie mit `elements` an:

```javascript
# in der Applikation: die Grundliste
client.elements+ { tag "Sidebar.Layers" }
client.elements+ { tag "Toolbar.Print" }

# im Projekt: um ein Werkzeug ergänzen
client.addElements+ { tag "Sidebar.Select" }
client.addElements+ { tag "Toolbar.Select" }
```

%warn
Ein Werkzeug ist erst benutzbar, wenn seine Client-Elemente eingetragen sind. Die zugehörige Server-Aktion zu aktivieren genügt nicht: Ohne Element erscheint kein Bedienelement in der Oberfläche, und die Funktion bleibt für den Nutzer unerreichbar.
%end

Jedes Element kann `permissions` tragen und damit auf bestimmte Rollen beschränkt werden – so sehen nur berechtigte Nutzer ein Werkzeug. Über `before` und `after` bestimmen Sie die Einfügeposition relativ zu einem anderen Element.

### Beispiel-Konfiguration ::

```javascript
client {
    elements [
        { tag "Sidebar.Layers" }
        { tag "Sidebar.Search" }
        { tag "Sidebar.Edit" permissions.read "allow editor, deny all" }
        { tag "Toolbar.Identify.Click" }
        { tag "Toolbar.Print" }
        { tag "Infobar.Position" }
        { tag "Infobar.Scale" }
        { tag "Infobar.ZoomIn" }
        { tag "Infobar.ZoomOut" }
    ]
}
```

`elements` legt die vollständige Elementliste einer typischen Projektoberfläche fest: Ebenenbaum und Suche in der Seitenleiste, Objektabfrage und Druck in der Werkzeugleiste sowie Positions-, Maßstabs- und Zoom-Anzeigen in der Infoleiste. Die Reihenfolge der Einträge ist die Reihenfolge in der Oberfläche. Das Element `Sidebar.Edit` trägt eigene `permissions` und erscheint daher nur für die Rolle `editor`. Die Container `Sidebar`, `Toolbar` und `Infobar` müssen nicht eigens aufgeführt werden.

## Elemente

Die folgenden Tabellen führen die Elemente auf, die der Client kennt, geordnet nach dem Bereich der Oberfläche, in dem sie erscheinen.

### Seitenleiste

| Tag | Funktion |
|---|---|
| `Sidebar.Layers` | Ebenenbaum |
| `Sidebar.Search` | Suche |
| `Sidebar.Overview` | Übersichtskarte |
| `Sidebar.Project` | Projektinformationen |
| `Sidebar.Edit` | Editieren |
| `Sidebar.Annotate` | Markierungen |
| `Sidebar.Dimension` | Bemaßung |
| `Sidebar.Select` | Auswahl |
| `Sidebar.Style` | Darstellung von Layern ändern |
| `Sidebar.Alkis` | Flurstückssuche (ALKIS) |
| `Sidebar.User` | An- und Abmeldung |
| `Sidebar.AccountAdmin` | Kontenverwaltung |

### Werkzeugleiste

| Tag | Funktion |
|---|---|
| `Toolbar.Identify.Click` | Objekt durch Klick abfragen |
| `Toolbar.Identify.Hover` | Objekt bei Mausbewegung abfragen |
| `Toolbar.Select` | Objekte auswählen |
| `Toolbar.Select.Draw` | Auswahl durch Zeichnen einer Fläche |
| `Toolbar.Edit` | Editieren |
| `Toolbar.Annotate.Draw` | Markierung zeichnen |
| `Toolbar.Dimension` | Bemaßen |
| `Toolbar.Print` | Drucken |
| `Toolbar.Screenshot` | Bildschirmfoto der Karte |
| `Toolbar.Lens` | Räumliche Suche mit verschiebbarer Lupe |
| `Toolbar.Location` | Eigenen Standort anzeigen |
| `Toolbar.Gekos` | GekoS-Werkzeug |

### Infoleiste

| Tag | Funktion |
|---|---|
| `Infobar.Position` | Koordinaten der Mausposition |
| `Infobar.Scale` | Maßstabsanzeige und -eingabe |
| `Infobar.Rotation` | Drehung der Karte |
| `Infobar.ZoomIn`, `Infobar.ZoomOut`, `Infobar.ZoomReset` | Zoom-Schaltflächen |
| `Infobar.Loader` | Ladeanzeige |
| `Infobar.About` | Info zur Anwendung |
| `Infobar.Help` | Hilfe |
| `Infobar.HomeLink` | Verweis auf die Startseite |
| `Infobar.Link`, `Infobar.LinkButton` | frei konfigurierbarer Verweis |
| `Infobar.Spacer` | Abstandhalter |

`Infobar.Link` und `Infobar.LinkButton` nehmen zusätzliche `options` entgegen – `title`, `href`, `className` sowie `target` mit den Werten `blank` (neues Fenster) und `frame`. Damit binden Sie eigene Verweise in die Infoleiste ein:

```javascript
client.addElements+ {
    tag "Infobar.LinkButton"
    options {
        title "Impressum"
        href "/impressum.html"
        target "blank"
    }
}
```

### Weitere Elemente

| Tag | Funktion |
|---|---|
| `Decoration.ScaleRuler` | Maßstabsleiste in der Karte |
| `Decoration.Attribution` | Quellenangabe in der Karte |
| `Altbar.Search` | Suchfeld in der oberen Leiste |
| `Task.*` | Einträge im Kontextmenü eines Objekts: `Task.Annotate`, `Task.Lens`, `Task.Search`, `Task.Select`, `Task.Zoom` |

Die Container – `Sidebar`, `Toolbar`, `Infobar` und die übrigen – müssen Sie nicht eigens aufführen: Der Client leitet sie aus dem Namensteil vor dem Punkt ab und legt sie selbst an. Es genügt, die enthaltenen Elemente zu nennen.

Die `Task`-Einträge sowie die internen `Shared.*`- und `Tool.*`-Elemente werden ohnehin immer geladen; sie in die Liste aufzunehmen ist wirkungslos.

%ref "gws.base.client.core.Config"
