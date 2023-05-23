# Client :/admin-de/config-az/client

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

Für mehr Info siehe [Web Server](/admin-de/config-az/web).

## UI-Konfiguration

%reference_de 'gws.base.client.core.Config'

Jedes GBD WebSuite Projekt, wie auch die Hauptanwendung, kann ein ``client`` Objekt enthalten, das verschiedene Optionen für den Client und dessen UI-Layout beschreibt, so dass Sie bestimmte UI-Elemente pro Projekt ein- und ausschalten können. Dieses Objekt besteht aus zwei Teilen: ``options`` (generelle Optionen) und ``elements`` (Auflistung der UI Elemente).

### options

Die Optionen sind wie folgt:

| Option | Typ | Bedeutung | Beispielwert |
|---|---|---|---|
|``infobarVisible`` | *bool* | untere Leiste ("Infobar") ist sichtbar | ``false`` |
|``sidebarActiveTab`` | *str* | actives Icon in der linken Leiste ("Sidebar") | ``"Sidebar.Layers"`` |
|``sidebarVisible`` | *bool* | Sidebar ursprünglich sichtbar | ``true`` |
|``sidebarSize`` | *int* | Anzahl von sichtbaren Icons in der Sidebar |  ``4`` |
|``toolbarSize`` | *int* | Anzahl von sichtbaren Icons in der Toolbar |  ``5`` |

### elements

Jede Element Konfiguration enthält einen Tag-Namen sowie optional eine Zugriffsberechtigung, sodass die UI Elemente nur für bestimmte Nutzer erscheinen.

Es werden folgende Element-Tags unterstützt:

*untere Leiste*

|Tag | Bedeutung|
|---|---|
|``Infobar.About`` | Link "Über uns" |
|``Infobar.Help`` | Link "Hilfe" |
|``Infobar.HomeLink`` | Link "Startseite" |
|``Infobar.Loader`` | Ladeanzeige |
|``Infobar.Position`` | Positionsanzeige |
|``Infobar.Rotation`` | Rotationsanzeige |
|``Infobar.Scale`` | Maßstabsanzeige |
|``Infobar.Spacer`` | flexibler Abstandhalter |

*linke Leiste*

|Tag | Bedeutung|
|---|---|
|``Sidebar.Alkis`` | Flurstücksuche |
|``Sidebar.Annotate`` | Markieren und Messen |
|``Sidebar.Bplan`` | Bauleitplanung, siehe [Bauleitplanung](/admin-de/config-az/bplan) |
|``Sidebar.Dimension`` | Bemaßung, siehe [Bemaßung](/admin-de/config-az/dimension) |
|``Sidebar.Edit`` | Digitalisierung, siehe [Digitalisierung](/admin-de/plugin/edit) |
|``Sidebar.Layers`` | Layerbaum |
|``Sidebar.Overview`` | Projektübersicht |
|``Sidebar.Search`` | Suche |
|``Sidebar.Select`` | Auswahl von Objekten |
|``Sidebar.User`` | Login und Logout |

*Werkzeuge*

|Tag | Bedeutung|
|---|---|
|``Toolbar.Annotate`` | Markieren und Messen |
|``Toolbar.Dimension`` | Bemaßung |
|``Toolbar.Dprocon`` | D-ProCon aufurfen, siehe [D-ProCon](/admin-de/config-az/dprocon) |
|``Toolbar.Gekos`` | GekoS aufrufen, siehe [GekoS Integration](/admin-de/config-az/gekos) |
|``Toolbar.Identify.Click`` | Objekt-Identifizierung mit Klicken |
|``Toolbar.Identify.Hover`` | Objekt-Identifizierung mit Ziehen |
|``Toolbar.Lens`` | räumliche Suche |
|``Toolbar.Location`` | aktueller Standout |
|``Toolbar.Print`` | Drucken |
|``Toolbar.Select`` | Objekte auswählen |
|``Toolbar.Snapshot`` | Screenshot |
|``Toolbar.Tabedit`` | tabellarisches Editieren, siehe [Tabellarisces Editieren](/admin-de/plugin/tabedit) |

*Popup-Menüs für Feature Objekte*

|Tag | Bedeutung|
|---|---|
|``Task.Annotate`` | Markieren und Messen |
|``Task.Lens`` | räumliche Suche |
|``Task.Search`` | im Objekt suchen |
|``Task.Select`` | Objekt auswählen |
|``Task.Zoom`` | zum Objekt hinzoomen |

*sonstiges*

|Tag | Bedeutung|
|---|---|
|``Altbar.Search`` | Such-Box in rechten Bereich |
|``Decoration.Attribution`` | Attribution auf der Karte |
|``Decoration.ScaleRuler`` | Maßstabsbalken auf der Karte |
|``Storage.Read`` | Datenablage lesen, siehe [Datenablage](/admin-de/config-az/storage) |
|``Storage.Write`` | in der Datenablage speichern, siehe [Datenablage](/admin-de/config-az/storage) |

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

%info
 Unter [Styling](/admin-de/config-az/style) finden Sie eine Auflistung von CSS Eingenschaften.
%end
