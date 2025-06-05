# relatedFeatureList :/admin-de/config/model/field/relatedFeatureList

%reference_de 'gws.plugin.model_field.related_feature_list.Config'

Ein Gegensatz zum Typ `relatedFeature`, beschreibt die "parent" Seite von einer `1:M` Beziehung. Als Widget wird `featureList` verwendet.

Die andere Seite der Beziehung muss mit `relatedFeature` konfiguriert werden.

    %dbgraph 'Eine Strasse hat mehrere Häuser.'
        house(id pk, street_id -> street.id)
        street(id pk)
    %end

    models+ {
        uid "modelStreet"
        fields+ {
            type "relatedFeatureList"
            name "houses"
            relationship { 
                modelUid "modelHouse" 
                fieldName "street"
            }
            widget.type "featureList"
        }
    }
    
    models+ {
        uid "modelHouse"
        fields+ {
            type "relatedFeature"
            name "street"
            relationship { 
                modelUid "modelStreet" 
                foreignKey "street_id"
            }
            widget.type "featureSelect"
        }
    }

