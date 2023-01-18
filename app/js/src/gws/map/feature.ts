import * as ol from 'openlayers';

import * as types from '../types';
import * as api from '../core/api';
import * as lib from '../lib';



export class Feature implements types.IFeature {
    uid: string = '';

    attributes: types.Dict = {};
    editedAttributes: types.Dict = {};
    category: string = '';
    elements: types.Dict = {};
    layer?: types.IFeatureLayer = null;
    model?: types.IModel = null;
    oFeature?: ol.Feature = null;
    cssSelector: string;

    isNew: boolean = false;
    isSelected: boolean = false;

    map: types.IMapManager;

    constructor(map) {
        this.map = map;
        this.uid = lib.uniqId('_feature_');
    }

    setProps(props) {
        this.model = this.map.app.models.getModel(props.modelUid);

        this.attributes = props.attributes || {};
        this.editedAttributes = {};

        let uid = this.attributes[this.keyName];
        if (uid)
            this.uid = String(uid);

        this.category = props.category || '';
        this.elements = props.elements || {};

        let layerUid = props.layerUid;
        if (layerUid)
            this.layer = this.map.getLayer(layerUid) as types.IFeatureLayer;


        this.isNew = Boolean(props.isNew);
        this.isSelected = Boolean(props.isSelected);

        let shape = this.attributes[this.geometryName];
        if (shape) {
            this.createOrUpdateOlFeature(this.map.shape2geom(shape));
        }

        return this;
    }

    //

    get keyName() {
        return this.model.keyName
    }

    get geometryName() {
        return this.model.geometryName
    }

    get geometry() {
        return this.oFeature ? this.oFeature.getGeometry() : null;
    }

    get shape() {
        let geom = this.geometry;
        return geom ? this.map.geom2shape(geom) : null;
    }

    get isFocused() {
        return this === this.map.focusedFeature;
    }

    get isDirty() {
        return !lib.isEmpty(this.editedAttributes);
    }

    getProps(depth) {
        return this.map.featureProps(this, depth);
    }

    getAttribute(name: string): any {
        return this.attributes[name];
    }

    getEditedAttribute(name: string): any {
        if (this.editedAttributes && (name in this.editedAttributes))
            return this.editedAttributes[name];
        return this.attributes[name];
    }

    //

    setGeometry(geom: ol.geom.Geometry) {
        this.createOrUpdateOlFeature(geom);
        this.attributes[this.geometryName] = this.map.geom2shape(geom);
        return this.redraw();
    }

    setNew(f: boolean) {
        this.isNew = f;
        return this.redraw();
    }

    setSelected(f: boolean) {
        this.isSelected = f;
        return this.redraw();
    }

    resetEdits() {
        this.editedAttributes = {};
    }

    commitEdits() {
        // this.originalAttributes = {...this.attributes};
    }

    redraw() {
        if (this.oFeature)
            this.oFeature.changed();
        return this;
    }

    //

    isSame(feature: types.IFeature) {
        return feature.layer === this.layer && feature.uid === this.uid;
    }

    updateFrom(feature: types.IFeature) {
        this.attributes = feature.attributes ? {...feature.attributes} : {};
        this.category = feature.category;
        this.elements = feature.elements ? {...feature.elements} : {};
        this.layer = feature.layer;
        this.model = feature.model;
        this.isNew = feature.isNew;
        this.isSelected = feature.isSelected;

        let shape = this.attributes[this.geometryName];
        if (shape) {
            this.createOrUpdateOlFeature(this.map.shape2geom(shape));
        }

        let uid = this.attributes[this.keyName];
        if (uid)
            this.uid = String(uid);

        this.redraw();
    }

    whenGeometryChanged() {

    }


    //

    protected createOrUpdateOlFeature(geom) {
        if (this.oFeature)
            this.oFeature.setGeometry(geom);
        else
            this.oFeature = new ol.Feature(geom);

        if (this.oFeature) {
            this.oFeature['_gwsFeature'] = this;
            this.oFeature.setStyle((f, r) => this.oStyleFunc(f, r))
        }

        this.oFeature.getGeometry().on('change', () => {
            let geom = this.oFeature.getGeometry();
            if (this.geometryName)
                this.attributes[this.geometryName] = this.map.geom2shape(geom);
            this.whenGeometryChanged()
        });

    }


    protected currentStyle() {
        let spec = '';

        if (this.isSelected) spec = 'isSelected';
        if (this.isNew) spec = 'isNew';
        if (this.isDirty) spec = 'isDirty';
        if (this.isFocused) spec = 'isFocused';

        if (this.cssSelector) {
            let c = this.cssSelector;
            if (spec)
                c += '.' + spec;
            let s = this.map.style.getFromSelector(c);
            if (s)
                return s;
        }

        if (this.layer && this.layer.cssSelector) {
            let c = this.layer.cssSelector;
            if (spec)
                c += '.' + spec;
            let s = this.map.style.getFromSelector(c);
            if (s)
                return s;
        }

        if (spec) {
            if (this.cssSelector) {
                let s = this.map.style.getFromSelector(this.cssSelector);
                if (s)
                    return s;
            }
            if (this.layer && this.layer.cssSelector) {
                let s = this.map.style.getFromSelector(this.cssSelector);
                if (s)
                    return s;
            }
        }

        let gt = this.oFeature.getGeometry().getType();
        let c = '.defaultGeometry_' + gt.toUpperCase();
        if (spec)
            c += '.' + spec;

        return this.map.style.getFromSelector(c);
    }

    protected oStyleFunc(oFeature, resolution) {
        let s = this.currentStyle();
        console.log('XXX', s)
        if (!s) {
            return [];
        }
        return s.apply(oFeature.getGeometry(), this.elements['label'], resolution);

    }

}
