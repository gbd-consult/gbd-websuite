GBD WebSuite Client
===================

Obwohl die GBD WebSuite als gewöhnlicher Webserver arbeiten kann, ist ihr Hauptzweck, zusammen mit einem "reichen" Javascript-Client verwendet zu werden, der in der Lage ist, dynamische Web-Maps wie OpenLayers of Leaflet anzuzeigen. Wir bieten einen solchen Client als Teil der GBD WebSuite an und stellen einige Optionen in der Serverkonfiguration zur Verfügung, um unseren Client gezielt zu unterstützen.

Projekt HTML Seite
------------------

Um Ihr Projekt in einem Webbrowser anzuzeigen, benötigen Sie eine HTML-Seite, die unseren Javascript-Client (s. doc: `client`) und die Projekt-ID enthalten sollte, damit der Client weiß, welches Projekt geladen werden soll. Auf der Seite muss sich ein div-Element mit dem Klassennamen ``gws`` befinden. Hier wird die Client-Benutzeroberfläche geladen. Ansonsten können Sie Ihre Startseite frei gestalten. Hier ist ein Beispiel ::

    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8"/>
        <title>My First Project!</title>

        <!-- CSS und Skripten von GWS Client
        <link rel="stylesheet" href="/gws-client/gws-light-6.1.css" type="text/css">
        <script src="/gws-client/gws-vendor-6.1.js"></script>
        <script src="/gws-client/gws-client-6.1.js"></script>

        <style>
            .gws {
                position: fixed;
                left: 10px;
                top: 20px;
                right: 40px;
                bottom: 50px;
            }
        </style>

        <!-- Project uid, as defined in the config file -->
        <script>
            GWS_PROJECT_UID = "project1";
        </script>

        <!-- Your own css, if needed -->
        <link rel="stylesheet" href="/my-style.css" type="text/css">

        <!-- Your additional css/scripts and other resources -->

    </head>

    <body>
        <!-- This is where the Client will be loaded -->
        <div class="gws"></div>

        You can add more content here...
    </body>
    </html>

Platzieren Sie diese Datei in Ihrem konfigurierten ``web``-Verzeichnis (s.: doc: "web"), um sie im Web zur Verfügung zu stellen.

UI-Konfiguration
----------------

Jedes GBD WebSuite Projekt, wie auch die Hauptanwendung, kann die ``client`` Konfiguration haben, die verschiedene Optionen für den Client enthält und dessen UI-Layout beschreibt, so dass Sie bestimmte UI-Elemente pro Projekt ein- und ausschalten können.

Beispiel für die Client-Konfiguration ::

    {

        ## root UI Element:

        "tag": "ui",

        "options": {
            ## Die aktive Seitenleiste ist "Layers"
            "sidebarActiveTab": "Sidebar.Layers",

            ## sidebar is intially visible
            "sidebarVisible": False,

            ## vorgewählte Infoleiste-Taste
            "toolbarActiveButton": "Toolbar.Identify.Click",
        },

        "elements": [

            ## Kartendekorationselemente:

            { "tag": "Decoration.ScaleRuler"},
            { "tag": "Decoration.Attribution"},

            ## Infoleiste (normalerweise am unteren Bildschirmrand)

            {
                "tag": "Infobar",
                "elements": [
                    {
                        ## auf der linken Seite der Infoleiste, Zoom-Werkzeuge anzeigen:

                        "tag": "Infobar.LeftSide",
                        "elements": [
                            {"tag": "Infobar.ZoomOut"},
                            {"tag": "Infobar.ZoomIn"},
                            {"tag": "Infobar.ZoomBox"},
                            {"tag": "Infobar.ZoomReset"},
                        ]
                    },
                    {

                        ## auf der rechten Seite der Infoleiste den "About"-Link anzeigen, der sich in einem Pop-Over-Frame öffnet:

                        "tag": "Infobar.RightSide",
                        "elements": [
                            {
                                "tag": "Infobar.Link",
                                "options": {
                                    "title": "About",
                                    "href": "https://example.org/about",
                                    "target": "frame"
                                }
                            },
                        ]
                    },

                ]
            },

            ## Symbolleiste (normalerweise oben rechts)

            {
                "tag": "Toolbar",
                "elements": [

                    ## Anzeige der Messwerkzeuge in der Symbolleiste

                    {
                        "tag": "Toolbar.Group",
                        "elements": [
                            {"tag": "Toolbar.Measure.Line"},
                            {"tag": "Toolbar.Measure.Polygon"},
                            {"tag": "Toolbar.Measure.Circle"},
                            {"tag": "Toolbar.Measure.Clear"},
                            {"tag": "Toolbar.Measure.Cancel"},
                        ]
                    },

                    ## Anzeige der Drucktaste in der Symbolleiste

                    {
                        "tag": "Toolbar.Group",
                        "elements": [
                            {"tag": "Toolbar.Print.Go"},
                        ]
                    },

                    ## Anzeige des Suchfeldes in der Symbolleiste

                    {"tag": "Toolbar.Search"},
                ]
            },

            ## Sidebar (normalerweise oben links)

            {
                "tag": "Sidebar",
                "elements": [

                    ## Anzeige der Registerkarte Ebenen
                    {"tag": "Sidebar.Layers"},

                    ## Anzeige der Registerkarte Suche
                    {"tag": "Sidebar.Search"},

                    ## Anzeige der Registerkarte Benutzer und Anmeldung
                    {"tag": "Sidebar.User"},
                ]
            }
        ]
    }

Layer flags
-----------

Neben der UI-Konfiguration kann jede Kartenebene eine Reihe von booleschen Optionen haben, die dem Client mitteilen, wie diese Ebene angezeigt werden soll. Siehe :ref:`server_admin_en_configref_gws_gis_layer_ClientOptions` für Details.
