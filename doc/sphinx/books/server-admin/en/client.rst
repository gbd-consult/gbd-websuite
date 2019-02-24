GBD WebSuite Client
===================

Although GBD WebSuite can work as an ordinary webserver, its primary purpose is to be used together with a "rich" javascript client, capable to display dynamic web maps, like OpenLayers of Leaflet. We offer such a client as a part of the GBD WebSuite, and provide a few options in the server configuration to support our client specifically.

UI Configuration
----------------

Each GBD WebSuite project, as well as the main application, can have the ``client`` configuration, which provides various options for the client and describes its UI layout, so that you can turn specific UI elements on and off on a per-project basis. Additionally, you can add ``access`` blocks to specific elements to turn them on and off for specific users only.

Example of the client configuration ::


    {

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

            {"tag": "Decoration.ScaleRuler"},
            {"tag": "Decoration.Attribution"},

            ## infobar (normally, at the bottom of the screen)

            {"tag": "Infobar.ZoomOut"},
            {"tag": "Infobar.ZoomIn"},
            {"tag": "Infobar.ZoomReset"},
            {"tag": "Infobar.Position"},
            {"tag": "Infobar.Scale"},
            {"tag": "Infobar.Loader"},
            {"tag": "Infobar.Spacer"},
            {"tag": "Infobar.HomeLink"},
            {"tag": "Infobar.Help"},
            {"tag": "Infobar.About"},

            ## client toolbar

            {"tag": "Toolbar.Identify.Click"},
            {"tag": "Toolbar.Lens"},
            {"tag": "Toolbar.Annotate.Draw"},
            {"tag": "Toolbar.Print"},
            {"tag": "Toolbar.Snapshot"},

            ## sidebar

            {"tag": "Sidebar.Layers"},
            {"tag": "Sidebar.Search"},
            {"tag": "Sidebar.Overview"},
            {"tag": "Sidebar.Annotate"},
            {"tag": "Sidebar.User"}
        ]


Layer flags
-----------

Besides the UI configuration, each map layer can have a set of boolean options, telling the GBD WebSuite Client how to display this layer. See :ref:`server_admin_en_configref_gws_gis_layer_ClientOptions` for details.
