# Aktion "search" :/admin-de/konfiguration/action/search

Die Aktion `search` führt räumliche und attributive Suchen über die konfigurierten Such-Provider aus und liefert die Treffer als Objekte an den Client. Mit `limit` begrenzen Sie die maximale Trefferzahl, mit `tolerance` die Standard-Suchtoleranz und mit `categories` die verfügbaren Suchkategorien.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "search"
    limit 500
    categories [
        "Interessante Orte"
        "Stadtbezirke"
        "Natur"
    ]
}

finders+ {
    type "postgres"
    tableName "edit.poi"
    category "Interessante Orte"
    models+ {
        type "postgres"
        fields+ { name "name" type "text" textSearch { type "any" minLength 1 } }
    }
}
```

`limit` begrenzt die Gesamtzahl der zurückgelieferten Treffer. `categories` legt die Kategorien fest, nach denen der Nutzer die Ergebnisliste filtern kann. Die eigentlichen Such-Provider konfigurieren Sie getrennt als `finders`; jeder Finder ordnet sich über sein Feld `category` einer der hier genannten Kategorien zu.

%ref "gws.base.search.action.Config"
%demo "search_multi"
%demo "search_teaser"
%demo "search_view"
