GBD WebSuite Client
===================

Obwohl die GBD WebSuite als gewöhnlicher Webserver arbeiten kann, ist ihr Hauptzweck, zusammen mit einem "reichen" Javascript-Client verwendet zu werden, der in der Lage ist, dynamische Web-Maps wie OpenLayers of Leaflet anzuzeigen. Wir bieten einen solchen Client als Teil der GBD WebSuite an und stellen einige Optionen in der Serverkonfiguration zur Verfügung, um unseren Client gezielt zu unterstützen.

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

Neben der UI-Konfiguration kann jede Kartenebene eine Reihe von booleschen Optionen haben, die dem Client mitteilen, wie diese Ebene angezeigt werden soll.
