GBD WebSuite Client
===================

Although GBD WebSuite can work as an ordinary webserver, its primary purpose is to be used together with a "rich" javascript client, capable to display dynamic web maps, like OpenLayers of Leaflet. We offer such a client as a part of the GBD WebSuite, and provide a few options in the server configuration to support our client specifically.

UI Configuration
----------------

Each GBD WebSuite project, as well as the main application, can have the ``client`` configuration, which provides various options for the client and describes its UI layout, so that you can turn specific UI elements on and off on a per-project basis.

Example of the client configuration ::


    {

        ## root UI element:

        "tag": "ui",

        "options": {
            ## active sidebar tab is "Layers"
            "sidebarActiveTab": "Sidebar.Layers",

            ## sidebar is intially visible
            "sidebarVisible": False,

            ## pre-selected infobar button
            "toolbarActiveButton": "Toolbar.Identify.Click",
        },

        "elements": [

            ## map decoration elements:

            { "tag": "Decoration.ScaleRuler"},
            { "tag": "Decoration.Attribution"},

            ## infobar (normally, at the bottom of the screen)

            {
                "tag": "Infobar",
                "elements": [
                    {
                        ## on the left side of the infobar, display zoom tools:

                        "tag": "Infobar.LeftSide",
                        "elements": [
                            {"tag": "Infobar.ZoomOut"},
                            {"tag": "Infobar.ZoomIn"},
                            {"tag": "Infobar.ZoomBox"},
                            {"tag": "Infobar.ZoomReset"},
                        ]
                    },
                    {

                        ## on the right side of the infobar, display the "about" link, which opens in a pop-over frame:

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

            ## toolbar (normally, at the top right)

            {
                "tag": "Toolbar",
                "elements": [

                    ## display measure tools in the toolbar

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


                    ## display the print button in the toolbar

                    {
                        "tag": "Toolbar.Group",
                        "elements": [
                            {"tag": "Toolbar.Print.Go"},
                        ]
                    },

                    ## display the search box in the toolbar

                    {"tag": "Toolbar.Search"},
                ]
            },

            ## sidebar (normally, at the top left)

            {
                "tag": "Sidebar",
                "elements": [

                    ## display the Layers tab
                    {"tag": "Sidebar.Layers"},

                    ## display the Search tab
                    {"tag": "Sidebar.Search"},

                    ## display the User and Login tab
                    {"tag": "Sidebar.User"},
                ]
            }
        ]
    }


Layer flags
-----------

Besides the UI configuration, each map layer can have a set of boolean options, telling the GBD WebSuite Client how to display this layer. See :ref:`server_admin_en_configref_gws_gis_layer_ClientOptions` for details.
