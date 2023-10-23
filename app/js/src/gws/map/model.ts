import * as ol from "openlayers";

import * as types from '../types';
import * as api from '../core/api';
import * as feature from './feature';
import * as lib from '../lib';
import {IFeature, IModel} from "../types";


export class ModelRegistry implements types.IModelRegistry {
    index: { [uid: string]: Model };
    app: types.IApplication;

    constructor(app: types.IApplication) {
        this.app = app;
        this.index = {};
        this.addModel({
            canCreate: false,
            canDelete: false,
            canRead: false,
            canWrite: false,
            isEditable: false,
            supportsKeywordSearch: false,
            supportsGeometrySearch: false,
            fields: [],
            geometryName: 'geometry',
            uidName: 'uid',
            loadingStrategy: api.core.FeatureLoadingStrategy.all,
            title: '',
            uid: '',
        });
    }


    readModel(props: api.base.model.Props) {
        return new Model(this, props);
    }

    addModel(props: api.base.model.Props) {
        let m = new Model(this, props);
        this.index[m.uid] = m;
        return m;
    }

    getModel(uid) {
        let m = this.index[uid || ''];
        if (!m)
            throw new Error(`model ${uid} not found`);
        return m;
    }

    defaultModel() {
        return this.index[''];
    }

    getModelForLayer(layer) {
        for (let m of Object.values(this.index))
            if (m.layerUid === layer.uid)
                return m
    }

    featureFromProps(props) {
        return this.getModel(props.modelUid).featureFromProps(props);
    }

    featureListFromProps(propsList) {
        let features = [];
        for (let props of propsList)
            features.push(this.featureFromProps(props));
        return features;
    }
}

interface FeatureMap {
    [uid: string]: types.IFeature
}

export class Model implements types.IModel {
    canCreate: boolean;
    canDelete: boolean;
    canRead: boolean;
    canWrite: boolean;
    isEditable: boolean;
    supportsKeywordSearch: boolean;
    supportsGeometrySearch: boolean;
    fields: Array<types.IModelField>;
    geometryCrs: string
    geometryName: string;
    geometryType: api.core.GeometryType
    uidName: string;
    layerUid: string;
    loadingStrategy: api.core.FeatureLoadingStrategy;
    title: string;
    uid: string;

    featureMap: FeatureMap;

    registry: ModelRegistry;

    constructor(registry, props: api.base.model.Props) {
        this.registry = registry;
        this.featureMap = {};

        this.canCreate = props.canCreate;
        this.canDelete = props.canDelete;
        this.canRead = props.canRead;
        this.canWrite = props.canWrite;
        this.isEditable = props.isEditable;
        this.supportsKeywordSearch = props.supportsKeywordSearch;
        this.supportsGeometrySearch = props.supportsGeometrySearch;
        this.geometryCrs = props.geometryCrs;
        this.geometryName = props.geometryName;
        this.geometryType = props.geometryType;
        this.uidName = props.uidName;
        this.layerUid = props.layerUid;
        this.loadingStrategy = props.loadingStrategy;
        this.title = props.title;
        this.uid = props.uid;

        this.fields = [];

        if (props.fields)
            for (let p of props.fields)
                this.fields.push(new ModelField(this).setProps(p));

        return this;
    }

    get layer() {
        return this.registry.app.map.getLayer(this.layerUid) as types.IFeatureLayer;
    }

    getField(name) {
        for (let f of this.fields)
            if (f.name === name)
                return f;
    }

    newFeature() {
        return new feature.Feature(this);
    }


    featureWithAttributes(attributes: types.Dict): types.IFeature {
        return this.newFeature().setAttributes(attributes);
    }

    featureFromGeometry(geom: ol.geom.Geometry): types.IFeature {
        let map = this.registry.app.map;
        return this.newFeature().setGeometry(geom);
    }

    featureFromOlFeature(oFeature: ol.Feature): types.IFeature {
        let map = this.registry.app.map;
        return this.newFeature().setOlFeature(oFeature);
    }

    featureFromProps(props: api.core.FeatureProps): types.IFeature {
        let map = this.registry.app.map;
        let attributes = props.attributes || {};

        for (let f of this.fields) {
            let val = attributes[f.name];

            if (val && f.attributeType === 'feature') {
                attributes[f.name] = this.registry.featureFromProps(val);
            }

            if (val && f.attributeType === 'featurelist') {
                attributes[f.name] = val.map(p => this.registry.featureFromProps(p));
            }
        }

        return this.newFeature().setProps({...props, attributes});
    }

    featureListFromProps(propsList: Array<api.core.FeatureProps>): Array<types.IFeature> {
        return propsList.map(props => this.featureFromProps(props));
    }

    featureAttributes(feature: types.IFeature, relDepth?: number): types.Dict {
        if (lib.isEmpty(this.fields)) {
            return feature.attributes;
        }

        let attributes = {};
        let depth = relDepth || 0;

        for (let f of this.fields) {
            let val = feature.attributes[f.name];

            switch (f.attributeType) {
                case 'feature':
                    if (val && depth > 0) {
                        attributes[f.name] = (val as types.IFeature).getProps(depth - 1);
                    }
                    break;
                case 'featurelist':
                    if (val && depth > 0) {
                        attributes[f.name] = val.map(f => (f as types.IFeature).getProps(depth - 1));
                    }
                    break;
                default:
                    if (val !== null && val !== undefined) {
                        attributes[f.name] = val;
                    }
            }
        }

        return attributes;
    }


    featureProps(feature: types.IFeature, relDepth?: number): api.core.FeatureProps {
        return {
            attributes: this.featureAttributes(feature, relDepth),
            cssSelector: feature.cssSelector,
            isNew: feature.isNew,
            modelUid: this.uid,
            uid: feature.uid,
            views: feature.views,
            createWithFeatures: feature.createWithFeatures.map(c => c.getMinimalProps())
        }
    }
}

export class ModelField implements types.IModelField {
    attributeType: api.core.AttributeType;
    geometryType: api.core.GeometryType;
    name: string;
    title: string;
    type: string;
    uid: string;

    relatedModelUids: Array<string>;
    widgetProps: api.ext.props.modelWidget;

    model: Model;

    constructor(model) {
        this.model = model;
    }

    setProps(props: api.ext.props.modelField): ModelField {
        this.attributeType = props.attributeType;
        this.geometryType = props.geometryType;
        this.name = props.name;
        this.relatedModelUids = props.relatedModelUids || [];
        this.title = props.title;
        this.uid = props.uid;
        this.widgetProps = props.widget;

        return this;
    }

    relatedModels() {
        return (this.relatedModelUids || []).map(uid => this.model.registry.getModel(uid));
    }

    addRelatedFeature(targetFeature: IFeature, relatedFeature: IFeature) {

        if (this.attributeType == api.core.AttributeType.feature) {
            targetFeature.editAttribute(this.name, relatedFeature);
        }

        if (this.attributeType == api.core.AttributeType.featurelist) {
            let curList = targetFeature.currentAttributes()[this.name] || [];
            let newList = [];
            let added = false;

            for (let f of curList) {
                if (f.model !== relatedFeature.model || f.uid !== relatedFeature.uid) {
                    newList.push(f);
                } else {
                    newList.push(relatedFeature);
                    added = true
                }
            }

            if (!added) {
                newList.push(relatedFeature);
            }

            targetFeature.editAttribute(this.name, newList);
        }
    }

    removeRelatedFeature(targetFeature: IFeature, relatedFeature: IFeature) {

        if (this.attributeType == api.core.AttributeType.feature) {
            targetFeature.editAttribute(this.name, null);
        }

        if (this.attributeType == api.core.AttributeType.featurelist) {
            let curList = targetFeature.currentAttributes()[this.name] || [];
            let newList = [];

            for (let f of curList) {
                if (f.model !== relatedFeature.model || f.uid !== relatedFeature.uid) {
                    newList.push(f);
                }
            }

            targetFeature.editAttribute(this.name, newList);
        }
    }


}
