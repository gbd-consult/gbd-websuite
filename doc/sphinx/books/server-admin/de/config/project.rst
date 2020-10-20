Projekte
========

^REF gws.common.project.Config

Ein *Projekt* (``project``) in der GBD WebSuite besteht aus einer Karte, Druckvorlagen und zusätzlichen Optionen. In Abschnitten ``api`` und ``client`` können Sie die im Hauptkonfig definierte Aktionen und Client-Optionen überschreiben bzw. erweitern. Mittels ``access`` können Sie die Zugriffsrechte zu Projekten steuern.

Eine Projektkonfiguration sollte mindestens  ``title`` und eine ``map`` enthalten. Eine minimale Projektkonfiguration kann z.B. wie folgt aussehen: ::

    {
        "title": "Hello",
        "uid": "project1",
        "map": {
            "extent": [554000, 6461000, 954000, 6861000],
            "scales": [1e3, 5e3, 1e4],
            "layers": [
                {
                    "title": "OpenStreetMap",
                    "type": "tile",
                    "url": "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"
                }
            ]
        }
    }

Projekt-Vorlagen
----------------

Im Projekt kann eine Info-Vorlage konfiguriert werden, die im Client gezeigt werden sobald der Nutzer die Projekt-Eigenschaften aktiviert. Diese Vorlage muss mit dem ``subject: project.description`` versehen werden.

^SEE Siehe ^format für mehr Info on Vorlagen.

Aktion ``projekt``
------------------

^REF gws.ext.action.project.Config

Mit dieser Aktion wird ein Projekt für den GWS Client freigeschaltet. Wenn diese Aktion fehlt, kann das Projekt nicht im Client aufgerufen werden, kann aber für andere Zwecke wie z.B. ein WMS Dienst verwender werden.
