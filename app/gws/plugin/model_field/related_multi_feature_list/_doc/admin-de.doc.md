# relatedMultiFeatureList :/admin-de/config/model/field/

%reference_de 'gws.plugin.model_field.related_multi_feature_list.Config'

Der Typ `relatedMultiFeatureList` beschreibt eine `1:M` Beziehung zwischen mehreren Modellen. In der Konfiguration müssen alle verknüpften Felder mit Namen aufgelistet werden.

Die andere Seite der Beziehung muss mit `relatedFeature` konfiguriert werden.

    %dbgraph 'Eine Strasse hat mehrere Objekte wie Laternen, Bushaltestellen oder Bäume.'
        street(id pk)
        lamp (id pk, street_id -> street.id)
        stop (id pk, street_id -> street.id)
        tree (id pk, street_id -> street.id)
    %end

    models+ {
        uid "modelStreet"
        fields+ {
            name "objekte"
            type  "relatedMultiFeatureList"
            relationships [
                { modelUid "modelLamp" fieldName "street" }
                { modelUid "modelStop" fieldName "street" }
                { modelUid "modelTree" fieldName "street" }
            ]
        }
    }
    models+ {
        uid "modelLamp"
        fields+ { 
            type "relatedFeature" 
            name "street"
            relationship { modelUid "modelStreet" foreignKey "street_id" }
        }
    }
    models+ {
        uid "modelStop"
        fields+ { 
            type "relatedFeature" 
            name "street"
            relationship { modelUid "modelStreet" foreignKey "street_id" }
        }
    }
    models+ {
        uid "modelTree"
        fields+ { 
            type "relatedFeature" 
            name "street"
            relationship { modelUid "modelStreet" foreignKey "street_id" }
        }
    }
