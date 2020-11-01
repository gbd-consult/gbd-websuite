Client
======

GWS Client ist eine Javascript Browser-Anwendung, die dafür konzipiert ist, zusammen mit dem GWS Server zu arbeiten.

HTML Vorlage
------------

Um Ihr Projekt in einem Webbrowser anzuzeigen, benötigen Sie eine HTML-Seite, die den Client und einige Projekt Informationen enthalten sollte, damit der Client weiß, welches Projekt geladen werden soll. Auf der Seite muss sich ein div-Element mit dem Klassennamen ``gws`` befinden. Hier wird die Client-Benutzeroberfläche geladen. Ansonsten können Sie Ihre Startseite frei gestalten.

Der Client selbst besteht aus drei Dateien:

- ``gws-light-<VERSION>.css`` - Style Datei
- ``gws-vendor-<VERSION>.js`` - Javascript Bibliothek
- ``gws-client-<VERSION>.js`` - Javascript Anwendung

Diese Dateien sind im GWS Server unter einer speziellen Adresse ``/gws-client`` erreichbar. Sie brauchen diese Adresse in Ihrer Konfiguration *nicht* explizit einzurichten.

Hier ist ein Vorlage der Client-Seite, die Sie nach Ihren Bedürfnissen anpassen können: ::

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

Um diese Vorlage für mehrere Projekte zu verwenden, ersetzen Sie den ``script`` Abschnitt mit dynamischen Werten: ::

    GWS_PROJECT_UID = "{project.uid}";
    GWS_LOCALE = "{project.locales[0]}";

und speichern Sie die Vorlage als z.B. "project.html" in Ihren ``assets`` Ordner. Dann kann diese Seit im Browser wie folgt aufgerufen werden: ::

    http://example.com/_?cmd=assetHttpGetPath&path=project.html&projectUid=meinprojekt

Diese URL kann auch "schöner" gemacht werden, indem Sie diese Rewrite-Regel verwenden: ::

    {
        "match": "^/project/([a-z][a-z0-9_]*)",
        "target": "_?cmd=assetHttpGetPath&&path=project.html&projectUid=$1"
    }

Dann heißt die URL einfach ::

    http://example.com/project/meinprojekt

Für mehr Info s. ^web.

UI-Konfiguration
----------------

^REF gws.common.client.Config

Jedes GBD WebSuite Projekt, wie auch die Hauptanwendung, kann ein ``client`` Objekt enthalten, das verschiedene Optionen für den Client und dessen UI-Layout beschreibt, so dass Sie bestimmte UI-Elemente pro Projekt ein- und ausschalten können. Dieses Objekt besteht aus zwei Teilen: ``options`` (generelle Optionen) und ``elements`` (Auflistung der UI Elemente).

options
~~~~~~~

Die Optionen sind wie folgt:

{TABLE head}
Option | Typ | Bedeutung | Beispielwert
``infobarVisible`` | *bool* | untere Leiste ("Infobar") ist sichtbar | ``false``
``sidebarActiveTab`` | *str* | actives Icon in der linken Leiste ("Sidebar") | ``"Sidebar.Layers``
``sidebarVisible`` | *bool* | Sidebar ursprünglich sichtbar | ``true``
``sidebarSize`` | *int* | Anzahl von sichtbaren Icons in der Sidebar |  ``4``
``toolbarSize`` | *int* | Anzahl von sichtbaren Icons in der Toolbar |  ``5``
{/TABLE}

elements
~~~~~~~~

Jede Element Konfiguration enthält einen Tag-Namen sowie optional eine Zugriffsberechtigung, sodass die UI Elemente nur für bestimmte Nutzer erscheinen.

Es werden folgende Element-Tags unterstützt:

{TABLE head}
Tag | Bedeutung
``Altbar.Search`` | ...
``Decoration.Attribution`` | ...
``Decoration.ScaleRuler`` | ...
``Infobar.About`` | ...
``Infobar.Help`` | ...
``Infobar.HomeLink`` | ...
``Infobar.Link`` | ...
``Infobar.Loader`` | ...
``Infobar.Position`` | ...
``Infobar.Rotation`` | ...
``Infobar.Scale`` | ...
``Infobar.Spacer`` | ...
``Sidebar.Alkis`` | ...
``Sidebar.Annotate`` | ...
``Sidebar.Bplan`` | ...
``Sidebar.Dimension`` | ...
``Sidebar.Edit`` | ...
``Sidebar.Layers`` | ...
``Sidebar.Overview`` | ...
``Sidebar.Search`` | ...
``Sidebar.Select`` | ...
``Sidebar.Style`` | ...
``Sidebar.UIDemo`` | ...
``Sidebar.User`` | ...
``Storage.Read`` | ...
``Storage.Write`` | ...
``Task.Annotate`` | ...
``Task.Lens`` | ...
``Task.Search`` | ...
``Task.Select`` | ...
``Task.Zoom`` | ...
``Toolbar.Annotate`` | ...
``Toolbar.Dimension`` | ...
``Toolbar.Dprocon`` | ...
``Toolbar.Gekos`` | ...
``Toolbar.Identify.Click`` | ...
``Toolbar.Identify.Hover`` | ...
``Toolbar.Lens`` | ...
``Toolbar.Location`` | ...
``Toolbar.Print`` | ...
``Toolbar.Select`` | ...
``Toolbar.Snapshot`` | ...
``Toolbar.Tabedit`` | ...
{/TABLE}

Layer flags
-----------

^REF gws.common.layer.types.ClientOptions

Neben der UI-Konfiguration kann jede Kartenebene eine Reihe von booleschen Optionen haben, die dem Client mitteilen, wie diese Ebene angezeigt werden soll. Siehe Referenz für Details.

CSS Anpassungen
---------------

Sie können den Stil der integrierten Funktionen anpassen, z. B. Markierungen von Suchergebnisse oder Messungen. Es gibt folgende vordefinierte CSS-Selektoren:

{TABLE head}
CSS Klasse | Funktion
``.gws.modMarkerFeature`` | Markierung für Suchergebnisse
``.gws.modAnnotatePoint`` | Punkt-Markierung
``.gws.modAnnotateLine`` | Linien-Markierung
``.gws.modAnnotatePolygon`` | Polygon-Markierung
``.gws.modAnnotateBox`` | Box-Markierung
``.gws.modAnnotateCircle`` | Kreis-Markierung
{/TABLE}

^SEE Unter ^style finden Sie eine Auflistung von CSS Eingenschaften.
