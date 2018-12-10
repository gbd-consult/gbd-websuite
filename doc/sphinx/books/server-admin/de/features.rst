Feature Transformation
=======================

Um ein reibungsloses Benutzererlebnis zu gewährleisten, bietet die GBD WebSuite verschiedene Werkzeuge zur Transformation und Neuformatierung von Feature-Daten aus verschiedenen Quellen. Wenn der GBD WebSuite Client ein Feature anzeigt, sucht er nach den folgenden Attributen und zeigt, falls diese vorhanden sind, eine schön formatierte Feature-Info-Box an:

TABLE
    *title* ~ Feature-Titel
    *shortText* ~ Kurzbeschreibung des Features
    *longText* ~ Detaillierte Beschreibung
    *imageUrl* ~ Illustration für das Feature
    *label* ~ Kartenbeschriftung für das Feature
/TABLE

Wenn ein Formatwert mit einem ``<`` beginnt, zeigt der GBD WebSuite Client ihn im HTML-Format an.

Sie können die Option ``meta`` verwenden, um unterschiedlich strukturierte Features neu zu formatieren, um ein einheitliches Aussehen zu erreichen. Betrachten wir zum Beispiel einen Layer "Stores", der auf einer WMS-Quelle basiert, die Feaure-Daten in folgendem Format liefert::

    name    -  Geschäftsname
    owner   - Name des Eigentümers
    address - Straße und Hausnummer
    photo   - ein Dateiname des Speicherbildes

Für diese Ebene könnte die Option ``meta`` wie folgt aussehen (beachten Sie die Verwendung von html):: 

    "meta": {
        "format": {
            "title": "Store {name}",
            "shortText": "<p>This store is run by <em>{owner}</em>. The address of the store is <strong>{address}</strong></p>",
            "imageUrl": "{photo}"
        }
    }

Neben Ebenen können auch ``meta``-Konfigurationen zu Suchanbietern hinzugefügt werden, um Suchergebnisse neu zu formatieren (siehe :doc:`search`). 
