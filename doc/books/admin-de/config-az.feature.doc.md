# Feature :/admin-de/config-az/feature

Die GBD WebSuite enthält Werkzeuge mit denen die Features, aus verschiedenen Quellen, einheitlich strukturiert und dargestellt werden können.

## Datenmodelle

%reference_de 'gws.ext.config.model'

*Felder* (``fields``) beschreiben wie Attribute eines Quell-Features transformiert werden müssen. Für jedes Attribut eines GBD WebSuite-Features können Sie eine Regel anlegen, mit der Sie für dieses Attribut folgendes definieren:

- eine Bezeichnung (``name``)
- einen Titel (``title``)
- einen Wert. Das kann Quell-Feature Attribut sein (``source``) oder einen festen Wert (``value``) oder eine Formatierungs-String mit ``{...}`` Platzhaltern, die mit Attributen der Quell-Feature ersetzt werden.

Zum Beispiel, wenn eine Postgres Tabelle ``user`` die Spalten ``first_name``, ``last_name`` und ``age`` enthält, können Sie so transformieren:

```javascript

map.layers+ {
    title "Test"
    type "postgres"
    tableName "edit.user"

    models+ {
        type "postgres"

        fields+ {
            name "age"
            type "text"
            title "Alter"
        }

        templates+ {
            subject "feature.title"
            type "html"
            text "{{first_name}} {{last_name}}"
        }
    }
}
```

TODO! -> Stimmt so nicht mehr in R8?
Außerdem können Sie angeben welche Attribute editierbar (``editable``) sind. Wenn Sie eine Editierfunktion verwenden (siehe [Digitalisierung](/admin-de/plugin/edit) und [Tabellarishces Editieren](/admin-de/plugin/tabedit)), werden nur editierbare Attribute eines Feature für Editieren freigeschaltet.

Seit der Version 7, besteht die Möglichkeit, den Attributen spezielle Editoren bzw Validierungsregel zuzuordnen. Ein Editor kann mit ``widget`` konfiguriert werden:

```javascript

map.layers+ {
    title "Test"
    type "postgres"
    tableName "edit.user"

    models+ {
        type "postgres"

        fields+ {
            name "comment"
            type "text"
            title "Kommentar"
            widget { type "textarea" height 150}
        }

        fields+ {
            name "role"
            type "text"
            title "Rolle"
            widget {    type "select"
                        items [
                            { text "Administrator" value "admin" }
                            { text "Nutzer" value "user" }
                            { text "Gast" value "guest" }
                        ]
                    }
        }
    }
}
```

Es werden folgende Editor-Typen unterstützt:

| OPTION | BEDEUTUNG |
|---|---|
|``string`` | HTML ``<input>`` Element |
|``int`` bzw ``float`` | HTML ``<input type=number>`` Element |
|``text`` | HTML ``<textarea>`` Element |
|``select`` bzw. ``combo`` | ``<select>`` Element, die Werte müssen mit ``items`` als eine Liste von Wert-Titel Paaren konfiguriert werden |
|``checkbox`` | ``<input type=checkbox>`` Element für boolesche Attributen |
|``date`` | Datums-Eingabefeld |

Wenn kein Editor konfiguriert ist, wird vom Server einen am besten geeigneten Typ gewählt.

TODO! validator "format" Beispiel einbauen!

Für die Validierungsregel kann eine Liste ``validators`` einem Attribut zugeordnet werden:

```javascript

map.layers+ {
    title "Test"
    type "postgres"
    tableName "edit.user"

    models+ {
        type "postgres"

        fields+ {
            name "comment"
            type "text"
            title "Kommentar"
            widget { type "textarea" height 150}
            validators+ {type "required" message "Diese Angabe wird benötigt."}
        }

        fields+ {
            name "role"
            type "text"
            title "Rolle"
            widget {    type "select"
                        items [
                            { text "Administrator" value "admin" }
                            { text "Nutzer" value "user" }
                            { text "Gast" value "guest" }
                        ]
                    }
            validators+ { type "required" message "Diese Angabe wird benötigt."}
        }
    }
}

TODO! Stimmt so nicht mehr!

Es werden folgende Regel unterstützt:

| Typ | Parameter | Bedeutung |
|---|---|---|
|``required`` | | der Wert darf nicht leer sein |
|``range`` | ``min`` und ``max`` | der Wert muss eine Zahl zwischen ``min`` und ``max`` sein |
|``length`` | ``min`` und ``max`` | die Länge des Stringwerts muss zwischen ``min`` und ``max`` sein |
|``regex`` | ``pattern`` | der Wert muss mit dem regulären Ausdruck übereinstimmen |

## Client-Vorlagen

Sie können Vorlagen (siehe [Vorlagen](/admin-de/config-az/tamplate)) Konfigurieren um die Features an verschiedenen Stellen im Client darzustellen. Die Vorlagen sind mit einem entsprechenden ``subject`` Wert zu versehen

| ``subject`` | Funktion |
|---|---|
|``feature.title`` | Feature-Titel |
|``feature.teaser`` | Kurzbeschreibung des Features, erscheint in der Autocomplete-Box beim Suchen |
|``feature.description`` | detaillierte Beschreibung, erscheint in der Info-Box |
|``feature.label`` | Kartenbeschriftung für das Feature |

Diese Vorlagen können für Layer (siehe [Layer](/admin-de/config-az/layer)) oder Suchprovider (siehe [Suche](/admin-de/config-az/search)) konfiguriert werden.

## XML Vorlagen

Für WMS/WFS Dienste besteht die Möglichkeit, für bestimmte Features eine angepasste XML Präsentation zu konfigurieren. Dazu erstellen Sie in der Konfiguration der jeweiligen Dienstes eine Vorlage mit dem ``subject`` ``ows.GetFeatureInfo`` (siehe [OWS](/admin-de/config-az/ows)).
