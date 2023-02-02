# Models :/dev-en/overview/models


## reading flow (non-db models)


    source_feature_list = query the source, return a list of `FeatureData` objects
    
    for each source_feature in source_feature_list
        feature = new feature
        
        for each field in model
            field.read_from_dict(feature, source_feature.attributes)
        
        for each field in model
            field.compute_value(feature, access=read)
        
        feature_list.append(feature)

    return feature_list


Later on, when the `feature_list` is about to be sent to the client, the following steps are taken

    for each feature in feature_list
        views = apply view templates to the feature

        attributes = {}
        for each field in model
            if user.can_read(field)
                attributes[field.name] = feature.attributes[field.name]


## reading flow (db models)

In db models, each model is mapped to an ORM class (`model.saRecordClass`)

    select_statement = ...
    for each field in model
        field.add_to_select(select_statement)

    object_list = execute select_statement
    
    for each object in object_list
        feature = new feature

        for each field in model
            field.read_from_object(feature, object)

        for each field in model
            field.compute_value(feature, access=read)

        feature_list.append(feature)

    return feature_list

## writing flow (db models)


    for each feature_props in feature_props_list
        feature = new feature
        
        for each field in model
            field.read_from_dict(feature, feature_props.attributes)
        
        for each field in model
            field.compute_value(feature, access=write)
        
        feature_list.append(feature)

    return feature_list
