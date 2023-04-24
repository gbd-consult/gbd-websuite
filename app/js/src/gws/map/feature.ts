import * as ol from 'openlayers';

import * as types from '../types';
import * as api from '../core/api';
import * as lib from '../lib';


export class Feature implements types.IFeature {
    uid: string = '';

    attributes: types.Dict = {};
    category: string = '';
    views: types.Dict = {};
    layer?: types.IFeatureLayer = null;
    oFeature?: ol.Feature = null;
    cssSelector: string;

    isNew: boolean = false;
    isSelected: boolean = false;

    model: types.IModel;
    map: types.IMapManager;

    _editedAttributes: types.Dict = {};

    constructor(model: types.IModel) {
        this.model = model;
        this.map = model.registry.app.map;
        this.uid = lib.uniqId('_feature_');
    }

    setProps(props) {
        this.setAttributes(props.attributes || {});

        this.category = props.category || '';
        this.views = props.views || {};

        let layerUid = props.layerUid;
        if (layerUid)
            this.layer = this.map.getLayer(layerUid) as types.IFeatureLayer;

        this.isNew = Boolean(props.isNew);
        this.isSelected = Boolean(props.isSelected);

        this.cssSelector = props.cssSelector || '';

        return this;
    }

    setAttributes(attributes) {
        this.attributes = attributes || {};
        this._editedAttributes = {};

        let uid = this.attributes[this.keyName];
        if (uid)
            this.uid = String(uid);

        let shape = this.attributes[this.geometryName];
        if (shape)
            this.setShape(shape);

        return this;
    }

    setGeometry(geom: ol.geom.Geometry) {
        return this.setShape(this.map.geom2shape(geom));
    }

    setStyle(style: types.IStyle) {
        this.cssSelector = style.cssSelector;
        return this.redraw();
    }

    setShape(shape: api.base.shape.Props) {
        this.oFeature = this.ensureOlFeature();
        this.attributes[this.geometryName] = shape;
        this.updateOlFeatureFromShape(shape);
        this.bindOlFeature();
        return this.redraw();
    }

    setOlFeature(oFeature) {
        this.oFeature = oFeature;
        this.updateShapeFromOlFeature();
        this.bindOlFeature();
        return this.redraw();
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
        return !lib.isEmpty(this._editedAttributes);
    }

    getProps(depth) {
        return this.model.featureProps(this, depth);
    }

    getAttribute(name, defaultValue = null) {
        if (name in this.attributes)
            return this.attributes[name];
        return defaultValue;
    }

    //

    editAttribute(name: string, newValue) {
        let currValue = this.attributes[name];
        if (currValue === newValue)
            delete this._editedAttributes[name];
        else
            this._editedAttributes[name] = newValue;
    }

    currentAttributes() {
        return {
            ...this.attributes,
            ...this._editedAttributes,
        }
    }

    commitEdits() {
        this.attributes = this.currentAttributes();
        this._editedAttributes = {};
    }

    resetEdits() {
        this._editedAttributes = {};
    }

    //


    setNew(f: boolean) {
        this.isNew = f;
        return this.redraw();
    }

    setSelected(f: boolean) {
        this.isSelected = f;
        return this.redraw();
    }

    redraw() {
        if (this.oFeature)
            this.oFeature.changed();
        return this;
    }

    //

    clone() {
        let f = new Feature(this.model)
        f.attributes = {...this.attributes}
        f.category = this.category;
        f.views = {...this.views};
        f.layer = this.layer;
        f.cssSelector = this.cssSelector
        f.isNew = this.isNew
        return f;
    }

    whenGeometryChanged() {

    }


    //

    protected bindOlFeature() {
        this.oFeature['_gwsFeature'] = this;
        this.oFeature.setStyle((f, r) => this.oStyleFunc(f, r))
        this.oFeature.getGeometry().on('change', () => {
            this.updateShapeFromOlFeature();
            this.whenGeometryChanged();
            this.redraw();

        });

    }

    protected ensureOlFeature() {
        return this.oFeature || new ol.Feature();
    }

    protected updateOlFeatureFromShape(shape) {
        this.oFeature.setGeometry(this.map.shape2geom(shape));
    }


    protected updateShapeFromOlFeature() {
        let geom = this.oFeature.getGeometry();
        if (this.geometryName)
            this.attributes[this.geometryName] = this.map.geom2shape(geom);
    }

    protected currentStyle() {
        let spec = '';

        if (this.isSelected) spec = 'isSelected';
        if (this.isNew) spec = 'isNew';
        if (this.isDirty) spec = 'isDirty';
        if (this.isFocused) spec = 'isFocused';

        return this.map.style.findFirst(
            [this.cssSelector, this.layer?.cssSelector, '.defaultFeatureStyle'],
            this.geometry?.getType(),
            spec
        )
    }

    protected oStyleFunc(oFeature, resolution) {
        let s = this.currentStyle();
        if (!s) {
            return [];
        }
        return s.apply(oFeature.getGeometry(), this.views['label'], resolution);

    }

}
