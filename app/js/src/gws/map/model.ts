import * as ol from "openlayers";

import * as types from '../types';
import * as api from '../core/api';
import * as feature from './feature';


export class ModelRegistry implements types.IModelRegistry {
    mdict: { [uid: string]: Model };
    app: types.IApplication;

    constructor(app: types.IApplication) {
        this.app = app;
        this.mdict = {};
        this.mdict[''] = new Model(this, {
            canCreate: false,
            canDelete: false,
            canRead: false,
            canWrite: false,
            fields: [],
            geometryName: 'geometry',
            keyName: 'uid',
            loadingStrategy: api.core.FeatureLoadingStrategy.all,
            uid: '',
        })
    }


    addModel(props: api.base.model.Props) {
        this.mdict[props.uid] = new Model(this, props);
    }

    model(uid) {
        return this.mdict[uid || ''];
    }

    defaultModel() {
        return this.mdict[''];
    }

    modelForLayer(layer) {
        for (let m of Object.values(this.mdict))
            if (m.layerUid === layer.uid)
                return m
    }

    editableModels() {
        let d: { [k: string]: types.IModel } = {}
        for (let m of Object.values(this.mdict)) {
            if (m.canCreate || m.canWrite || m.canDelete)
                if (!d[m.layerUid])
                    d[m.layerUid] = m;
        }
        return Object.values(d)
    }

    featureFromProps(props) {
        return this.model(props.modelUid).featureFromProps(props);
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
    fields: Array<types.IModelField>;
    geometryCrs: string
    geometryName: string;
    geometryType: api.core.GeometryType
    keyName: string;
    layerUid: string;
    loadingStrategy: api.core.FeatureLoadingStrategy;
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
        this.geometryCrs = props.geometryCrs;
        this.geometryName = props.geometryName;
        this.geometryType = props.geometryType;
        this.keyName = props.keyName;
        this.layerUid = props.layerUid;
        this.loadingStrategy = props.loadingStrategy;
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

    get title() {
        let la = this.layer;
        return la ? la.title : '...';

    }

    getField(name) {
        for (let f of this.fields)
            if (f.name === name)
                return f;
    }

    featureWithAttributes(attributes: types.Dict): types.IFeature {
        let map = this.registry.app.map;
        return new feature.Feature(this, map).setAttributes(attributes);
    }

    featureFromGeometry(geom: ol.geom.Geometry): types.IFeature {
        let map = this.registry.app.map;
        return new feature.Feature(this, map).setGeometry(geom);
    }

    featureFromProps(props: api.core.FeatureProps): types.IFeature {
        let map = this.registry.app.map;
        let attributes = props.attributes || {};

        for (let f of this.fields) {
            let val = attributes[f.name];

            if (val && f.attributeType === 'feature') {
                attributes[f.name] = this.featureFromProps(val);
            }

            if (val && f.attributeType === 'featurelist') {
                attributes[f.name] = val.map(p => this.featureFromProps(p));
            }
        }

        return new feature.Feature(this, map).setProps({...props, attributes});
    }

    featureListFromProps(propsList: Array<api.core.FeatureProps>): Array<types.IFeature> {
        return propsList.map(props => this.featureFromProps(props));
    }

    featureProps(feature: types.IFeature, relationDepth?: number): api.core.FeatureProps {

        let atts = {};
        let depth = relationDepth || 0;

        if (feature.model) {

            for (let f of feature.model.fields) {
                let val = feature.attributes[f.name];

                switch (f.attributeType) {
                    case 'feature':
                        if (val && depth > 0) {
                            atts[f.name] = this.featureProps(val, depth - 1);
                        }
                        break;
                    case 'featurelist':
                        if (val && depth > 0) {
                            atts[f.name] = val.map(f => this.featureProps(f, depth - 1));
                        }
                        break;
                    default:
                        if (val !== null && val !== undefined) {
                            atts[f.name] = val;
                        }
                }
            }
        } else {
            atts = feature.attributes || {};
        }

        // let style = self.style.at(f.styleNames.normal);

        return {
            attributes: atts,
            views: {},
            // layerUid: feature.layer ? feature.layer.uid : null,
            modelUid: feature.model ? feature.model.uid : null,
            uid: feature.uid,
            isNew: feature.isNew,
            keyName: feature.keyName,
            geometryName: feature.geometryName,
            errors: [],
            // style: style ? style.props : null,
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

    relations: Array<types.IModelRelation>;
    widgetProps: api.ext.props.modelWidget;

    model: Model;

    constructor(model) {
        this.model = model;

    }

    setProps(props: api.ext.props.modelField): ModelField {
        this.attributeType = props.attributeType;
        this.geometryType = props.geometryType;
        this.name = props.name;
        this.relations = props.relations || [];
        this.title = props.title;
        this.uid = props.uid;
        this.widgetProps = props.widget;

        return this;
    }
}
