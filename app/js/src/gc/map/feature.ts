import * as ol from 'openlayers';

import {gws} from '../gws';
import * as types from '../types';
import * as lib from '../lib';


export class Feature implements types.IFeature {
    attributes: types.Dict = {};
    category: string = '';
    cssSelector: string;
    layer?: types.IFeatureLayer = null;
    views: types.Dict = {};

    oFeature?: ol.Feature = null;

    isNew: boolean = false;
    isSelected: boolean = false;

    model: types.IModel;
    map: types.IMapManager;

    uidName: string;
    geometryName: string;

    createWithFeatures = [];

    _editedAttributes: types.Dict = {};

    constructor(model: types.IModel) {
        this.model = model;
        this.map = model.registry.app.map;

        this.attributes = {};
        this.category = '';
        this.cssSelector = '';
        this.views = {};
        this.createWithFeatures = [];

        this.uidName = this.model.uidName;
        this.geometryName = this.model.geometryName;

        this.attributes[this.uidName] = lib.uniqId('_feature_');
    }

    get uid() {
        let s = this.attributes[this.uidName];
        return lib.isEmpty(s) ? '' : String(s);
    }

    setProps(props) {
        this.category = props.category || '';
        this.views = props.views || {};

        let layerUid = props.layerUid;
        if (layerUid)
            this.layer = this.map.getLayer(layerUid) as types.IFeatureLayer;

        this.isNew = Boolean(props.isNew);
        this.isSelected = Boolean(props.isSelected);

        this.cssSelector = props.cssSelector || '';

        this.uidName = props.uidName || this.model.uidName;
        this.geometryName = props.geometryName || this.model.geometryName;

        this.setAttributes(props.attributes || {});

        if (props.uid) {
            if (!this.attributes[this.uidName]) {
                this.attributes[this.uidName] = props.uid;
            }
        }

        return this;
    }

    setAttributes(attributes) {
        this.attributes = attributes || {};
        this._editedAttributes = {};

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

    setShape(shape: gws.base.shape.Props) {
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

    getProps(depth = 0) {
        return this.model.featureProps(this, depth);
    }

    getMinimalProps() {
        let p = this.model.featureProps(this, 0);
        return {
            attributes: {
                [this.model.uidName]: p.attributes[this.model.uidName],
            },
            cssSelector: '',
            isNew: p.isNew,
            modelUid: p.modelUid,
            uid: p.uid,
            views: {},
        }
    }

    getAttribute(name, defaultValue = null) {
        if (name in this.attributes)
            return this.attributes[name];
        return defaultValue;
    }

    getAttributeWithEdit(name, defaultValue = null) {
        if (name in this._editedAttributes)
            return this._editedAttributes[name];
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
        f.copyFrom(this);
        return f;
    }

    copyFrom(f: Feature) {
        this.attributes = {...f.attributes}
        this.category = f.category;
        this.views = {...f.views};
        this.layer = f.layer;
        this.cssSelector = f.cssSelector
        this.isNew = f.isNew
        this.createWithFeatures = f.createWithFeatures
    }

    whenGeometryChanged() {

    }

    whenSaved(f) {

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
