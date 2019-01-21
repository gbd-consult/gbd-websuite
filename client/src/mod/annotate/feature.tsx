import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as types from './types';

export const defaultLabelTemplates = {
    Point: '{x}, {y}',
    Line: '{len}',
    Polygon: '{area}',
    Circle: '{radius}',
    Box: '{w} x {h}',
};

export class Feature extends gws.map.Feature implements types.Feature {
    app: gws.types.IApplication;
    labelTemplate: string;
    selected: boolean;
    selectedStyle: gws.types.IMapStyle;
    shapeType: string;
    cc: any;

    get dimensions() {
        return computeDimensions(this.shapeType, this.geometry, this.map.projection);

    }

    get formData(): types.FeatureFormData {
        let dims = this.dimensions;
        return {
            x: formatCoordinate(this, dims.x),
            y: formatCoordinate(this, dims.y),
            radius: formatLengthForEdit(dims.radius),
            w: formatLengthForEdit(dims.w),
            h: formatLengthForEdit(dims.h),
            labelTemplate: this.labelTemplate
        }
    }


    constructor(app: gws.types.IApplication, args: types.FeatureArgs) {
        super(app.map, args);
        this.app = app;
        this.selected = false;
        this.labelTemplate = args.labelTemplate;
        this.selectedStyle = args.selectedStyle;
        this.shapeType = args.shapeType;

        this.oFeature.setStyle((oFeature, r) => {
            let s = this.selected ? this.selectedStyle : this.style;
            return s.apply(oFeature.getGeometry(), this.label, r);
        });
        this.oFeature.on('change', e => this.onChange(e));
        this.geometry.on('change', e => this.onChange(e));
        this.redraw();
    }

    setSelected(sel) {
        this.selected = sel;
        this.oFeature.changed();
    }

    onChange(e) {
        this.redraw();
    }

    redraw() {
        let master = this.app.controller(types.MASTER) as types.MasterController;
        let dims = this.dimensions;
        this.setLabel(formatTemplate(this, this.labelTemplate, dims));
        master.featureUpdated(this);
    }

    updateFromForm(ff: types.FeatureFormData) {
        this.labelTemplate = ff.labelTemplate;

        let t = this.shapeType;

        if (t === 'Point') {
            let g = this.geometry as ol.geom.Point;
            g.setCoordinates([
                Number(ff.x) || 0,
                Number(ff.y) || 0,

            ]);
        }

        if (t === 'Circle') {
            let g = this.geometry as ol.geom.Circle;

            g.setCenter([
                Number(ff.x) || 0,
                Number(ff.y) || 0,

            ]);
            g.setRadius(Number(ff.radius) || 1)
        }

        if (t === 'Box') {
            // NB: x,y = top left
            let g = this.geometry as ol.geom.Polygon;
            let x = Number(ff.x) || 0,
                y = Number(ff.y) || 0,
                w = Number(ff.w) || 100,
                h = Number(ff.h) || 100;
            let coords: any = [
                [x, y - h],
                [x + w, y - h],
                [x + w, y],
                [x, y],
                [x, y - h],
            ];
            g.setCoordinates([coords])
        }

        this.geometry.changed();
    }

}

function computeDimensions(shapeType, geom, projection) {

    if (shapeType === 'Point') {
        let g = geom as ol.geom.Point;
        let c = g.getCoordinates();
        return {
            x: c[0],
            y: c[1]
        }
    }

    if (shapeType === 'Line') {
        return {
            len: ol.Sphere.getLength(geom, {projection})
        }
    }

    if (shapeType === 'Polygon') {
        let g = geom as ol.geom.Polygon;
        let r = g.getLinearRing(0);
        return {
            len: ol.Sphere.getLength(r, {projection}),
            area: ol.Sphere.getArea(g, {projection}),
        };
    }

    if (shapeType === 'Box') {

        let g = geom as ol.geom.Polygon;
        let r = g.getLinearRing(0);
        let c: any = r.getCoordinates();

        // NB: x,y = top left
        return {
            len: ol.Sphere.getLength(r, {projection}),
            area: ol.Sphere.getArea(g, {projection}),
            x: c[0][0],
            y: c[3][1],
            w: Math.abs(c[1][0] - c[0][0]),
            h: Math.abs(c[2][1] - c[1][1]),
        }
    }

    if (shapeType === 'Circle') {
        let g = geom as ol.geom.Circle;
        let p = ol.geom.Polygon.fromCircle(g, 64, 0);
        let c = g.getCenter();

        return {
            ...computeDimensions('Polygon', p, projection),
            x: c[0],
            y: c[1],
            radius: g.getRadius(),
        }
    }

    return {};
}

function formatLengthForEdit(n) {
    return (Number(n) || 0).toFixed(0)
}

function formatCoordinate(feature: Feature, n) {
    return feature.map.formatCoordinate(Number(n) || 0);

}

function formatTemplate(feature: Feature, text, dims) {

    function _element(key) {
        let n = dims[key];
        if (!n)
            return '';
        switch (key) {
            case 'area':
                return _area(n);
            case 'len':
            case 'radius':
            case 'w':
            case 'h':
                return _length(n);
            case 'x':
            case 'y':
                return formatCoordinate(feature, n);

        }
    }

    function _length(n) {
        if (!n || n < 0.01)
            return '';
        if (n >= 1e3)
            return (n / 1e3).toFixed(2) + ' km';
        if (n > 1)
            return n.toFixed(0) + ' m';
        return n.toFixed(2) + ' m';
    }

    function _area(n) {
        let sq = '\u00b2';

        if (!n || n < 0.01)
            return '';
        if (n >= 1e5)
            return (n / 1e6).toFixed(2) + ' km' + sq;
        if (n > 1)
            return n.toFixed(0) + ' m' + sq;
        return n.toFixed(2) + ' m' + sq;
    }

    return (text || '').replace(/{(\w+)}/g, ($0, key) => _element(key));
}






