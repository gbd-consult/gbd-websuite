# Modell-Feld "relatedMultiFeatureList" :/admin-de/konfiguration/modelField/relatedMultiFeatureList

Das Feld `relatedMultiFeatureList` bildet eine 1:N-Beziehung von einem übergeordneten Modell zu mehreren untergeordneten Modellen ab: Der Wert des Feldes ist die Liste der Kindobjekte aus allen beteiligten Tabellen. Unter `related` geben Sie die verknüpften Modelle mit jeweils `toModel` und der Fremdschlüsselspalte `toColumn` an; mit `fromColumn` bestimmen Sie die Schlüsselspalte in dieser Tabelle (standardmäßig deren Primärschlüssel).

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "objekte"
    type "relatedMultiFeatureList"
    title "Objekte"
    related [
        { toModel "model_kanalarbeiten" toColumn "baustelle_id" }
        { toModel "model_sperrung"      toColumn "baustelle_id" }
        { toModel "model_umleitung"     toColumn "baustelle_id" }
    ]
}
```

Das Feld `objekte` fasst 1:M-Beziehungen zu mehreren Zieltabellen zusammen. Jeder Eintrag unter `related` benennt ein Zielmodell (`toModel`) und dessen Rückverweis-Spalte (`toColumn`), sodass Kanalarbeiten, Sperrungen und Umleitungen gemeinsam an der Baustelle bearbeitet werden.

%ref "gws.plugin.model_field.related_multi_feature_list.Config"
%demo "field_related_multi_feature_list"
