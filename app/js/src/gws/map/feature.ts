import * as ol from 'openlayers';

import * as types from '../types';
import * as api from '../core/gws-api';
import * as tools from '../tools';

export class Feature implements types.IMapFeature {
    uid: string = '';
    attributes: Array<api.Attribute> = [];
    elements: types.Dict = {};
    layerUid: string = '';
    mode: types.FeatureMode;

    styleNames = {
        normal: '',
        selected: '',
        edit: '',
    };

    label: string = '';
    oFeature?: ol.Feature = null;

    map: types.IMapManager;

    constructor(map, args: types.IMapFeatureArgs) {
        this.map = map;

        if (args.props) {
            this.uid = args.props.uid;
            this.attributes = args.props.attributes || [];
            this.elements = args.props.elements || {};
            this.layerUid = args.props.layerUid || '';
        }

        this.uid = this.uid || tools.uniqId(this.layerUid || '_feature_');

        let oFeature = this.oFeatureFromArgs(args);
        if (oFeature) {
            oFeature['_gwsFeature'] = this;
        }

        this.oFeature = oFeature;

        this.setStyles({
            normal: args.style || (args.props && args.props.style),
            selected: args.selectedStyle,
            edit: args.editStyle,
        });

        if (args.label) {
            this.setLabel(args.label);
        } else if (this.elements.label) {
            this.setLabel(this.elements.label);
        } else if (oFeature) {
            this.setLabel(oFeature.get('label'));
        }

        this.setMode('normal');
    }

    get geometry() {
        return this.oFeature ? this.oFeature.getGeometry() : null;
    }

    get shape() {
        let geom = this.geometry;
        return geom ? this.map.geom2shape(geom) : null;
    }

    getProps() {
        let style = this.map.style.at(this.styleNames.normal);
        return {
            attributes: this.attributes,
            elements: this.elements,
            layerUid: this.layerUid,
            shape: this.shape,
            style: style ? style.props : null,
            uid: this.uid
        }
    }

    getAttribute(name: string): any {
        for (let a of this.attributes) {
            if (a.name === name)
                return a.value;
        }
    }

    setMode(mode) {
        this.mode = mode;
        this.updateOlStyle();
    }

    setStyles(src) {
        this.styleNames = this.map.style.getMap(src);
        this.updateOlStyle()
    }

    setLabel(label: string) {
        this.label = label;
        if (this.oFeature)
            this.oFeature.set('label', label);
    }

    setGeometry(geom) {
        if (this.oFeature)
            this.oFeature.setGeometry(geom);
        else
            this.oFeature = new ol.Feature(geom);
    }

    setChanged() {
        if (this.oFeature)
            this.oFeature.changed();
    }

    protected updateOlStyle() {
        if (!this.oFeature) {
            return;
        }

        let currentStyleName = this.styleNames[this.mode];

        if (!currentStyleName) {
            this.oFeature.setStyle(null);
            return;
        }

        this.oFeature.setStyle((oFeature: ol.Feature, resolution: number) => {
            let sty = this.map.style.at(currentStyleName);
            return sty ? sty.apply(oFeature.getGeometry(), oFeature.get('label'), resolution) : [];
        });
    }

    protected oFeatureFromArgs(args) {
        if (args.oFeature) {
            return args.oFeature;
        }

        if (args.geometry) {
            return new ol.Feature(args.geometry)
        }

        if (args.props && args.props.shape) {
            let geom = this.map.shape2geom(args.props.shape);
            return new ol.Feature(geom);
        }
    }
}