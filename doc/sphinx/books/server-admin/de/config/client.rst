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
        "target": "_?cmd=assetHttpGetPath&path=project.html&projectUid=$1"
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

{TABLE}
Tag | Bedeutung
*untere Leiste*:
``Infobar.About`` | Link "Über uns"
``Infobar.Help`` | Link "Hilfe"
``Infobar.HomeLink`` | Link "Startseite"
``Infobar.Loader`` | Ladeanzeige
``Infobar.Position`` | Positionsanzeige
``Infobar.Rotation`` | Rotationsanzeige
``Infobar.Scale`` | Maßstabsanzeige
``Infobar.Spacer`` | flexibler Abstandhalter
*linke Leiste*:
``Sidebar.Alkis`` | Flurstücksuche
``Sidebar.Annotate`` | Markieren und Messen
``Sidebar.Bplan`` | Bauleitplanung (s. ^bplan)
``Sidebar.Dimension`` | Bemaßung (s. ^dimension)
``Sidebar.Edit`` | Digitalisierung (s. ^digitize)
``Sidebar.Layers`` | Layerbaum
``Sidebar.Overview`` | Projektübersicht
``Sidebar.Search`` | Suche
``Sidebar.Select`` | Auswahl von Objekten
``Sidebar.User`` | Login und Logout
*Werkzeuge*:
``Toolbar.Annotate`` | Markieren und Messen
``Toolbar.Dimension`` | Bemaßung
``Toolbar.Dprocon`` | DPro-Con aufurfen (s. ^dprocon)
``Toolbar.Gekos`` | GekoS aufrufen (s. ^gekos)
``Toolbar.Identify.Click`` | Objekt-Identifizierung mit Klicken
``Toolbar.Identify.Hover`` | Objekt-Identifizierung mit Ziehen
``Toolbar.Lens`` | räumliche Suche
``Toolbar.Location`` | aktueller Standout
``Toolbar.Print`` | Drucken
``Toolbar.Select`` | Objekte auswählen
``Toolbar.Snapshot`` | Screenshot
``Toolbar.Tabedit`` | tabellarisches Editieren (s. ^tabedit)
*Popup-Menüs für Feature Objekte*:
``Task.Annotate`` | Markieren und Messen
``Task.Lens`` | räumliche Suche
``Task.Search`` | im Objekt suchen
``Task.Select`` | Objekt auswählen
``Task.Zoom`` | zum Objekt hinzoomen
*sonstiges*:
``Altbar.Search`` | Such-Box in rechten Bereich
``Decoration.Attribution`` | Attribution auf der Karte
``Decoration.ScaleRuler`` | Maßstabsbalken auf der Karte
``Storage.Read`` | Datenablage lesen (s. ^storage)
``Storage.Write`` | in der Datenablage speichern (s. ^storage)
{/TABLE}

Layer flags
-----------

^REF gws.common.layer.types.ClientOptions

Neben der UI-Konfiguration kann jede Kartenebene eine Reihe von booleschen Optionen haben, die dem Client mitteilen, wie diese Ebene angezeigt werden soll.

{TABLE}
``exclusive`` | (bei Gruppenlayern) nur ein Unterlayer zeigen
``expanded`` | (bei Gruppenlayern) Gruppe im Layerbaum ausklappen
``listed`` | Layer im Layerbaum zeigen
``selected`` | Layer im Layerbaum auswählen
``unfolded`` | (bei Gruppenlayern) nur die Unterlayer zeigen, nicht die Gruppe selbst
``visible`` | Layer ist sichtbar
{/TABLE}

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
