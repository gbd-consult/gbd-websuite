# Client :/admin-de/config/client

Der GBD WebSuite Client ist eine Javascript Browser-Anwendung, die dafür konzipiert ist, zusammen mit dem GBD WebSuite Server zu arbeiten.

## HTML Vorlage

Um Ihr Projekt in einem Webbrowser anzuzeigen, benötigen Sie eine HTML-Seite, die den Client und einige Projekt Informationen enthalten sollte, damit der Client weiß, welches Projekt geladen werden soll. Auf der Seite muss sich ein div-Element mit dem Klassennamen ``gws`` befinden. Hier wird die Client-Benutzeroberfläche geladen. Ansonsten können Sie Ihre Seite frei gestalten.

Der Client selbst besteht aus drei Dateien:

- ``gws-light-<VERSION>.css`` - Style Datei
- ``gws-vendor-<VERSION>.js`` - Javascript Bibliothek
- ``gws-client-<VERSION>.js`` - Javascript Anwendung

Diese Dateien sind im GWS Server unter einer speziellen Adresse ``/gws-client`` erreichbar. Sie brauchen diese Adresse in Ihrer Konfiguration *nicht* explizit einzurichten.

Hier ist ein Vorlage der Client-Seite, die Sie nach Ihren Bedürfnissen anpassen können: ::

```html
    <!DOCTYPE html>
    <html>
    <head>
        <!-- Charset muss immer UTF8 sein! -->
        <meta charset="UTF-8"/>

        <!-- Für Mobilgeräte soll die Anwendung nicht skalierbar sein -->
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"/>

        <!-- CSS vom GWS Client -->
        <link rel="stylesheet" href="/gws-client/gws-light-6.1.1.css" type="text/css">

        <!-- "gws" Element kann frei positioniert werden, wir empfehlen "postion:absolute" bzw "fixed" -->
        <style>
            .gws {
                position: fixed;
                left: 10px;
                top: 20px;
                right: 40px;
                bottom: 50px;
            }
        </style>

        <!-- Hier muss die ID von Ihrem Projekt stehen, sowie die Sprache von Client Meldungen -->
        <script>
            GWS_PROJECT_UID = "meinprojekt";
            GWS_LOCALE = "de_DE";
        </script>

    </head>

    <body>
        <!-- In diesem Element wird der Client geladen. -->
        <div class="gws"></div>

        <!-- Javascript vom GWS Client -->
        <script src="/gws-client/gws-vendor-6.1.1.js"></script>
        <script src="/gws-client/gws-client-6.1.1.js"></script>

    </body>
    </html>
```

Um diese Vorlage für mehrere Projekte zu verwenden, setzen Sie den ``GWS_PROJECT_UID`` auf einen dynamischen Wert:

    GWS_PROJECT_UID = "{project.uid}";

und speichern Sie die Vorlage als z.B. "project.html" in Ihren ``assets`` Ordner. Dann kann diese Seit im Browser wie folgt aufgerufen werden:

    http://example.com/_?cmd=assetHttpGetPath&path=project.html&projectUid=meinprojekt

Diese URL kann auch "schöner" gemacht werden, indem Sie diese Rewrite-Regel verwenden: ::

```javascript
{
    "match": "^/project/([a-z][a-z0-9_]*)",
    "target": "_?cmd=assetHttpGetPath&path=project.html&projectUid=$1"
}
```

Dann heißt die URL einfach

    http://example.com/project/meinprojekt

Für mehr Info siehe [Web Server](/admin-de/config/web).

## UI-Konfiguration

%reference_de 'gws.base.client.core.Config'

Jedes GBD WebSuite Projekt, wie auch die Hauptanwendung, kann ein ``client`` Objekt enthalten, das verschiedene Optionen für den Client und dessen UI-Layout beschreibt, so dass Sie bestimmte UI-Elemente pro Projekt ein- und ausschalten können. Dieses Objekt besteht aus zwei Teilen: ``options`` (generelle Optionen) und ``elements`` (Auflistung der UI Elemente).

### Optionen

Die Optionen sind wie folgt:

| Option | Typ | Bedeutung | Beispielwert |
|---|---|---|---|
|``infobarVisible`` | *bool* | untere Leiste ("Infobar") ist sichtbar | ``false`` |
|``sidebarActiveTab`` | *str* | actives Icon in der linken Leiste ("Sidebar") | ``"Sidebar.Layers"`` |
|``sidebarVisible`` | *bool* | Sidebar ursprünglich sichtbar | ``true`` |
|``sidebarSize`` | *int* | Anzahl von sichtbaren Icons in der Sidebar |  ``4`` |
|``toolbarSize`` | *int* | Anzahl von sichtbaren Icons in der Toolbar |  ``5`` |

### Elemente

Jede Element Konfiguration enthält einen Tag-Namen sowie optional eine Zugriffsberechtigung, sodass die UI Elemente nur für bestimmte Nutzer erscheinen.

#### Grundlegende Konfiguration

```javascript
client.elements [
   # Werkzeugleiste
   { tag "Toolbar.Screenshot" }
   { tag "Toolbar.IdentifyClick" }
   ...   
   # Seitenleiste
   { tag "Sidebar.Layers" }
   { tag "Sidebar.Overview" }
   ...
   # Infoleiste
   { tag "Infobar.ZoomOut" }
   { tag "Infobar.ZoomIn" }
   ...
]
```

*Positionierung*

Mit ```after``` und ```before``` kann die Position in der Infobar gesteuert werden:

```
client.elements [
   { tag "Infobar.Scale" }
   {
       tag "Infobar.Link"
       after "Infobar.Scale"
   }
   {
       tag "Infobar.ButtonLink"
       before "Infobar.About"
   }
]
```

*Zugriffsbeschränkung*

```
Mit ```permissions``` können Zugriffsbeschränkungen definiert werden:

client.elements [
   {
       tag "Infobar.Link"
       permissions {
           # Berechtigungskonfiguration
       }
   }
]
```

Es werden folgende Element-Tags unterstützt:

**Übersicht Infoleiste**

|Tag | Bedeutung|
|---|---|
|``Infobar.ZoomOut`` | Rauszoomen |
|``Infobar.ZoomIn`` | Reinzoomen |
|``Infobar.ZoomReset`` | Zoomstufe zurücksetzen |
|``Infobar.About`` | Link "Über uns" |
|``Infobar.Help`` | Link "Hilfe" |
|``Infobar.Link`` | Freier Link als Text |
|``Infobar.ButtonLink`` | Freier Link als Icon |
|``Infobar.HomeLink`` | Link "Startseite" |
|``Infobar.Loader`` | Ladeanzeige |
|``Infobar.Position`` | Positionsanzeige |
|``Infobar.Rotation`` | Rotationsanzeige |
|``Infobar.Scale`` | Maßstabsanzeige |
|``Infobar.Spacer`` | flexibler Abstandhalter |


#### Beispiel Infobar.Link und Infobar.ButtonLink

Die GBD WebSuite bietet zwei UI-Elemente für die Infobar (untere Leiste der Kartenanwendung), um benutzerdefinierte Links zu integrieren:
  
  - **Infobar.Link** - Freier Link als Text
  - **Infobar.ButtonLink** - Freier Link als Icon

Diese Elemente ergänzen die vordefinierten Links wie `Infobar.About`, `Infobar.Help` und `Infobar.HomeLink` und ermöglichen die Integration zusätzlicher externer Verweise.

*Infobar.Link*

```javascript
client.elements+ {
    tag "Infobar.Link"
    options {
        title "Link Text"
        href "https://example.com"
        target "blank"  # Optional, möglich sind "blank" od. "frame"
    }
}
```

*Infobar.ButtonLink*

```javascript
client.elements+ {
    tag "Infobar.LinkButton"
    options {
        title "Tooltip für den Button"
        url "https://example.com"
        target "blank"  # Optional, möglich sind "blank" od. "frame"
        className "meine-css-klasse"
    }
}
```

Die CSS Klasse muss dann in "styles.css" (oder einer anderen CSS Datei) definiert werden, z.B.:

```css
.meine-css-klasse {
    background-image: url('/path/to/icon.svg');
}
```

**Übersicht Seitenleiste**

|Tag | Bedeutung|
|---|---|
|``Sidebar.Alkis`` | Flurstücksuche |
|``Sidebar.Annotate`` | Markieren und Messen |
|``Sidebar.Bplan`` | Bauleitplanung, siehe [Bauleitplanung](/admin-de/config/bplan) |
|``Sidebar.Dimension`` | Bemaßung, siehe [Bemaßung](/admin-de/plugin/dimension) |
|``Sidebar.Edit`` | Digitalisierung, siehe [Digitalisierung](/admin-de/plugin/edit) |
|``Sidebar.Layers`` | Layerbaum |
|``Sidebar.Overview`` | Projektübersicht |
|``Sidebar.Search`` | Suche |
|``Sidebar.Select`` | Auswahl von Objekten |
|``Sidebar.User`` | Login und Logout |

**Übersicht Werkzeugleiste**

|Tag | Bedeutung|
|---|---|
|``Toolbar.Annotate`` | Markieren und Messen |
|``Toolbar.Dimension`` | Bemaßung |
|``Toolbar.Dprocon`` | D-ProCon aufurfen, siehe [D-ProCon](/admin-de/config/dprocon) |
|``Toolbar.Gekos`` | GekoS aufrufen, siehe [GekoS Integration](/admin-de/config/gekos) |
|``Toolbar.Identify.Click`` | Objekt-Identifizierung mit Klicken |
|``Toolbar.Identify.Hover`` | Objekt-Identifizierung mit Ziehen |
|``Toolbar.Lens`` | räumliche Suche |
|``Toolbar.Location`` | aktueller Standout |
|``Toolbar.Print`` | Drucken |
|``Toolbar.Select`` | Objekte auswählen |
|``Toolbar.Screenshot`` | Screenshot |
|``Toolbar.Tabedit`` | tabellarisches Editieren, siehe [Tabellarisches Editieren](/admin-de/plugin/tabedit) |

**Übersicht Popup-Menüs für Feature Objekte**

|Tag | Bedeutung|
|---|---|
|``Task.Annotate`` | Markieren und Messen |
|``Task.Lens`` | räumliche Suche |
|``Task.Search`` | im Objekt suchen |
|``Task.Select`` | Objekt auswählen |
|``Task.Zoom`` | zum Objekt hinzoomen |

**Sonstiges**

|Tag | Bedeutung|
|---|---|
|``Altbar.Search`` | Such-Box in rechten Bereich |
|``Decoration.Attribution`` | Attribution auf der Karte |
|``Decoration.ScaleRuler`` | Maßstabsbalken auf der Karte |
|``Storage.Read`` | Datenablage lesen, siehe [Datenablage](/admin-de/config/datenablage) |
|``Storage.Write`` | in der Datenablage speichern, siehe [Datenablage](/admin-de/config/datenablage) |

### Layer flags

%reference_de 'gws.base.layer.core.ClientOptions'

Neben der UI-Konfiguration kann jede Kartenebene eine Reihe von booleschen Optionen haben, die dem Client mitteilen, wie diese Ebene angezeigt werden soll.

|OPTION|BEDEUTUNG|
|---|---|
|``exclusive`` | (bei Gruppenlayern) nur ein Unterlayer zeigen |
|``expanded`` | (bei Gruppenlayern) Gruppe im Layerbaum ausklappen |
|``unlisted`` | Layer im Layerbaum verstecken |
|``selected`` | Layer im Layerbaum auswählen |
|``unfolded`` | (bei Gruppenlayern) nur die Unterlayer zeigen, nicht die Gruppe selbst |
|``hidden`` | Layer ist unsichtbar |

## CSS Anpassungen

Sie können den Stil der integrierten Funktionen anpassen, z. B. Markierungen von Suchergebnisse oder Messungen. Es gibt folgende vordefinierte CSS-Selektoren:

|CSS Klasse | Funktion |
|---|---|
|``.gws.modMarkerFeature`` | Markierung für Suchergebnisse |
|``.gws.modAnnotatePoint`` | Punkt-Markierung |
|``.gws.modAnnotateLine`` | Linien-Markierung |
|``.gws.modAnnotatePolygon`` | Polygon-Markierung |
|``.gws.modAnnotateBox`` | Box-Markierung |
|``.gws.modAnnotateCircle`` | Kreis-Markierung |


Es besteht zudem die Möglichkeit den Stil einzelner von der WebSuite mitgelieferter HTML-Tags zu bearbeiten.
Hierfür können Sie eigene CSS-Klassen definieren, welche dann die mitgelieferten Eigenschaften überschreiben.

Solche CSS-Klassen können wie folgt ergänzt werden:

 - in einer existierenden .css-Datei (in /data/web)
 - in einer neuen .css-Datei, die dann im /data/web Verzeichnis hinterlegt wird und in der /data/assets/project.cx.html eingebunden werden muss
 - direkt an der Stelle in einem ``` <style> ```-Tag hinterlegt:
 ```css
 <style>
.myclass { color: red; }
</style>
<h1 class="myclass">test</h1>
```

 - Es ist ebenfalls möglich direkt an dem HTML-Element Inline Styles zu ergänzen:
 ```css
<h1 style="color: red;">test</h1>
```

Um zu sehen welche Werte bestimmten CSS Eigenschaften eines Elementes aktuell zugewiesen sind kann man sich im Browser Details zu einem Element anzeigen lassen. Dafür geht man per Rechtsklick auf das Element und wählt ``Untersuchen``.



%info
 Unter [Styling](/admin-de/config/style) finden Sie eine Auflistung von CSS Eigenschaften.
%end



 <!-- Dieser Teil muss noch zugeordnet werden -->
## Client-Vorlagen mit Features/Modellen konfigurieren

Sie können Vorlagen (siehe [Vorlagen](/admin-de/config/template)) Konfigurieren um Features bzw. [Modelle](/admin-de/config/models) an verschiedenen Stellen im Client darzustellen. Die Vorlagen sind mit einem entsprechenden ``subject`` Wert zu versehen

| ``subject`` | Funktion |
|---|---|
|``feature.title`` | Feature-Titel |
|``feature.teaser`` | Kurzbeschreibung des Features, erscheint in der Autocomplete-Box beim Suchen |
|``feature.description`` | detaillierte Beschreibung, erscheint in der Info-Box |
|``feature.label`` | Kartenbeschriftung für das Feature |

Diese Vorlagen können für Layer (siehe [Layer](/admin-de/config/layer)) oder Suchprovider (siehe [Suche](/admin-de/config/suche)) konfiguriert werden.
