# relatedFeature :/admin-de/config/model/field/relatedFeature

%reference_de 'gws.plugin.model_field.related_feature.Config'

Das Feld `relatedFeature` beschreibt die "child" Seite von einer `1:M` oder "parent-child" Beziehung. In der Konfiguration muss das "parent" Modell, sowie der Fremdschlüssel angegeben werden. Als Widgets für dieses Feld sind `featureSelect` oder `featureSuggest` geeignet.

    %dbgraph 'Häuser gehören zu einer Strasse.'
        house(id int pk, ..., street_id int fk)
        street(id int pk, ...)
        house.street_id >- street.id
    %end

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
    models+ {
        uid "modelStreet"
        ...
    }
