import * as types from '../types';
import * as lib from '../lib';
import * as api from '../core/api';


export class ModelRegistry implements types.IModelRegistry {
    models: { [uid: string]: Model };
    map: types.IMapManager;

    constructor(map: types.IMapManager) {
        this.map = map;
    }


    setProps(props: Array<api.base.model.Props>): ModelRegistry {
        this.models = {};

        for (let p of props) {
            this.models[p.uid] = new Model(this);
        }

        for (let p of props) {
            this.models[p.uid].setProps(p);
        }
        return this;
    }

    getModel(uid) {
        return this.models[uid];
    }

    getModelForLayer(layer) {
        for (let m of Object.values(this.models))
            if (m.layerUid === layer.uid)
                return m
    }

}

export class Model implements types.IModel {
    fields: Array<types.IModelField>;
    geometryName: string;
    keyName: string;
    layerUid: string;
    uid: string;
    registry: ModelRegistry;

    constructor(registry) {
        this.registry = registry;
    }

    setProps(props: api.base.model.Props): Model {
        this.geometryName = props.geometryName;
        this.keyName = props.keyName;
        this.layerUid = props.layerUid;
        this.uid = props.uid;

        this.fields = [];

        if (props.fields)
            for (let p of props.fields)
                this.fields.push(new ModelField(this).setProps(p));

        return this;
    }

    getLayer() {
        return this.registry.map.getLayer(this.layerUid) as types.IFeatureLayer;
    }

    getField(name) {
        for (let f of this.fields)
            if (f.name === name)
                return f;
    }

}

export class ModelField implements types.IModelField {
    name: string;
    type: string;
    attributeType: api.core.AttributeType;
    geometryType: api.core.GeometryType;
    title: string;

    relations: Array<types.IModelRelation>;
    // widget?: api.base.model.wid;

    model: Model;

    constructor(model) {
        this.model = model;

    }

    setProps(props: api.base.model.field.Props): ModelField {
        this.name = props.name;
        this.attributeType = props.attributeType;
        this.geometryType = props.geometryType;
        this.title = props.title;
        // this.widget = props.widget;


        this.relations = [];

        // if (props.relations) {
        //     for (let r of props.relations) {
        //         this.relations.push({
        //             type: r.type,
        //             model: this.model.registry.getModel(r.modelUid),
        //             fieldName: r.fieldName,
        //             title: r.title,
        //         })
        //     }
        // }

        return this;
    }
}
