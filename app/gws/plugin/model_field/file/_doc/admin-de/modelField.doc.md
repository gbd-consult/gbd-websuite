# Modell-Feld "file" :/admin-de/konfiguration/modelField/file

Das Feld `file` bildet ein Attribut mit einer Datei ab, etwa einem Dokument oder Bild, das an ein Objekt angehängt wird. Über `contentColumn` legen Sie fest, in welcher Datenbankspalte der Dateiinhalt gespeichert wird, alternativ verweist `pathColumn` auf eine im Dateisystem abgelegte Datei; mit `nameColumn` bestimmen Sie die Spalte für den Dateinamen. Es muss entweder `contentColumn` oder `pathColumn` gesetzt sein, und das Modell benötigt einen Primärschlüssel.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "file"
    type "file"
    title "Datei"
    contentColumn "content"
    nameColumn "filename"
}
```

Das Feld `file` verwaltet eine Datei innerhalb der Modelltabelle. `contentColumn` benennt die Spalte mit dem Dateiinhalt, `nameColumn` die Spalte mit dem Dateinamen.

%ref "gws.plugin.model_field.file.Config"
%demo "field_file"
%demo "field_file_list"
