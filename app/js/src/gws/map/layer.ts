import * as ol from 'openlayers';

import * as types from '../types';
import * as lib from '../lib';
import * as api from '../core/api';

const DEFAULT_TILE_TRANSITION = 700;

export class Layer implements types.ILayer {
    type = '';
    uid = '';
    title = '';
    attributes = {};

    props: api.base.layer.Props;

    parent: types.ILayer = null;
    children = [];

    expanded = false;
    visible = false;
    selected = false;
    listed = false;
    unfolded = false;
    exclusive = false;

    checked = false;

    editAccess = null;

    resolutions = [];
    minResolution = 0;
    maxResolution = 1e20;
    extent: ol.Extent;

    opacity: number;
    computedOpacity: number;

    attribution = null;
    description = null;

    displayMode = '';

    map: types.IMapManager = null;

    get oFeatures() {
        return [];
    }

    get oLayer() {
        return null;
    }

    get printPlane() {
        return null;
    }

    get inResolution() {
        let vs = this.map.viewState;
        return this.minResolution <= vs.resolution && vs.resolution <= this.maxResolution;
    }

    get shouldDraw() {
        return this.visible && this.inResolution;
    }

    get shouldList() {
        if (!this.listed)
            return false;
        if (!this.inResolution)
            return false;
        if (this.hasChildren)
            return this.children.some(la => la.shouldList);
        return true;
    }

    get hasChildren() {
        return this.children.length > 0;
    }

    get isSystem() {
        return String(this.uid)[0] === '_';
    }

    constructor(map: types.IMapManager, props: api.base.layer.Props) {
        this.map = map;
        this.props = props;

        this.type = this.props.type;
        this.title = this.props.title;
        this.uid = this.props.uid;
        this.attribution = this.props.metadata ? this.props.metadata.attribution : '';
        // this.editAccess = this.props.editAccess;

        this.displayMode = this.props.displayMode;

        this.resolutions = props.resolutions || this.map.resolutions;
        this.extent = props.extent || this.map.extent;
        this.opacity = props.opacity || 1;

        this.minResolution = Math.min(...this.resolutions);
        this.maxResolution = Math.max(...this.resolutions);

        let defaultOpts = {
            expanded: false,
            visible: true,
            selected: false,
            listed: true,
            unfolded: false,
            exclusive: false,
        };
        let opts = Object.assign(defaultOpts, props.clientOptions || {});

        this.expanded = opts['expanded'];
        this.visible = opts['visible'];
        this.selected = opts['selected'];
        this.listed = opts['listed'];
        this.unfolded = opts['unfolded'];
        this.exclusive = opts['exclusive'];

        this.checked = this.visible;

    }

    hide() {
    }

    show() {
    }

    changed() {
    }

    beforeDraw() {

    }

    reset() {

    }

    forceUpdate() {

    }

    setComputedOpacity(v) {
        this.computedOpacity = v;
    }
}

abstract class OlBackedLayer<T extends ol.layer.Layer> extends Layer {
    _oLayer: T;

    abstract createOLayer(): T;

    get printPlane(): api.base.printer.Plane {
        return {
            type: api.base.printer.PlaneType.raster,
            opacity: this.computedOpacity,
            layerUid: this.uid,
        }
    }

    get oLayer(): T {
        if (!this._oLayer) {
            let la = this.createOLayer();
            la.setVisible(true);
            la.on('change', () => {
                this.map.changed()
            });
            this._oLayer = la;
        }
        return this._oLayer;

    }

    reset() {
        this._oLayer = null;
    }

    hide() {
        this.oLayer.setVisible(false);
    }

    show() {
        this.oLayer.setVisible(true);
    }

    changed() {
        if (this.oLayer)
            this.oLayer.changed();
    }

    setComputedOpacity(v) {
        this.computedOpacity = v;
        this.oLayer.setOpacity(v);
    }

    forceUpdate() {
        if (this.oLayer && this.oLayer.getSource())
            this.oLayer.getSource().changed()
    }
}

export class BoxLayer extends OlBackedLayer<ol.layer.Image> {

    async loadImage(oImage: ol.Image, url: string) {
        let blob = await this.map.app.server.queueLoad(this.uid, url, 'blob');
        if (blob) {
            let img: any = oImage.getImage();
            img.src = window.URL.createObjectURL(blob);
        }
    }

    createOLayer() {
        return new ol.layer.Image({
            extent: this.extent,
            source: new ol.source.ImageWMS({
                url: this.props.url,
                ratio: 1,
                imageLoadFunction: (img, url) => this.loadImage(img, url),
                projection: this.map.projection,
                params: {
                    layers: ''
                },
            })
        });
    }
}

export class TileLayer extends OlBackedLayer<ol.layer.Image> {
    createOLayer() {
        return new ol.layer.Tile({
            source: new ol.source.TileImage({
                url: this.props.url,
                projection: this.map.projection,
                //transition: DEFAULT_TILE_TRANSITION,
                tileGrid: new ol.tilegrid.TileGrid({
                    extent: this.props.grid.extent,
                    tileSize: this.props.grid.tileSize,
                    resolutions: this.props.grid.resolutions,
                })
            })
        });
    }
}

export class XYZLayer extends OlBackedLayer<ol.layer.Image> {
    createOLayer() {
        return new ol.layer.Tile({
            source: new ol.source.XYZ({
                url: this.props.url,
                crossOrigin: 'Anonymous',

            })
        });
    }
}

export class TreeLayer extends OlBackedLayer<ol.layer.Image> {
    leaves: Array<string> = [];

    async loadImage(oImage: ol.Image, url: string) {
        let blob = await this.map.app.server.queueLoad(this.uid, url, 'blob');
        if (blob) {
            let img: any = oImage.getImage();
            img.src = window.URL.createObjectURL(blob);
        }
    }

    get shouldDraw() {
        return this.visibleLeavesUids().length > 0;
    }

    get printPlane() {
        let ls = this.visibleLeavesUids();

        if (!ls.length)
            return null;

        return {
            type: api.base.printer.PlaneType.raster,
            opacity: this.computedOpacity,
            layerUid: this.uid,
            subLayers: ls,
        }
    }

    beforeDraw() {
        let ls = this.visibleLeavesUids();

        if (ls.join() !== this.leaves.join()) {
            this.leaves = ls;
            (this._oLayer.getSource() as ol.source.ImageWMS).updateParams({
                layers: ls
            })
        }
    }

    createOLayer() {
        return new ol.layer.Image({
            extent: this.extent,
            source: new ol.source.ImageWMS({
                url: this.props.url,
                ratio: 1,
                imageLoadFunction: (img, url) => this.loadImage(img, url),
                projection: this.map.projection,
                params: {
                    layers: ''
                },
            })
        });
    }

    protected visibleLeavesUids() {
        return this.map
            .collect(this, la => la.type === 'leaf' && la.shouldDraw)
            .map(la => la.uid);
    }
}

export class LeafLayer extends Layer {
}

export class RootLayer extends Layer {
    get shouldList() {
        return true;
    }
}

export class GroupLayer extends Layer {
}


interface FeatureMap {
    [uid: string]: types.IFeature
}

export class FeatureLayer extends OlBackedLayer<ol.layer.Vector> implements types.IFeatureLayer {
    geometryType: string = '';
    cssSelector: string = '';
    fMap: FeatureMap = {};
    loadingStrategy: api.core.FeatureLoadingStrategy;

    lastBbox: string;
    loadState: string;

    get oFeatures() {
        let src = this.source;
        return src ? src.getFeatures() : [];
    }

    get source() {
        return this.oLayer ? this.oLayer.getSource() : null;
    }

    constructor(map, props) {
        super(map, props);

        this.cssSelector = props.cssSelector;

        if (props.geometryType)
            this.geometryType = props.geometryType;

        this.loadingStrategy = this.props.loadingStrategy;
        this.loadState = '';
        this.lastBbox = '';
    }

    printStyle() {
        let c,
            style,
            geom = this.geometryType ?'.' + this.geometryType.toLowerCase() : '';

        c = this.cssSelector;

        if (c) {
            style = this.map.style.getFromSelector(c + geom) || this.map.style.getFromSelector(c)
            if (style)
                return style

        }

        c = '.defaultFeatureStyle'
        return this.map.style.getFromSelector(c + geom) || this.map.style.getFromSelector(c)
    }

    get printPlane(): api.base.printer.Plane {
        let fs = lib.compact(this.features.map(f => f.getProps()));

        if (fs.length === 0)
            return null;

        let style = this.printStyle()

        return {
            type: this.props.url ? api.base.printer.PlaneType.vector : api.base.printer.PlaneType.features,
            opacity: this.computedOpacity,
            features: this.props.url ? [] : fs,
            style: style ? style.props : null,
            layerUid: this.uid,
        };
    }

    get features() {
        return Object.values(this.fMap);
    }


    createOLayer() {
        return new ol.layer.Vector({
            source: this.createSource(),
        });
    }

    protected createSource() {
        return new ol.source.Vector({});
    }

    forceUpdate() {
        this.lastBbox = '';
        if (this.oLayer && this.oLayer.getSource())
            this.oLayer.getSource().changed()
    }

    beforeDraw() {

        if (!this.props.url)
            return;

        if (this.loadingStrategy === api.core.FeatureLoadingStrategy.bbox) {
            let bbox = String(this.map.bbox);
            if (this.lastBbox === bbox) {
                return;
            }
            this.lastBbox = bbox;

            console.log('Vector:load', this.uid, this.loadingStrategy, 'reload', bbox);

            this.loadState = 'bbox_loading';
            this.loader();
        }

        if (this.loadingStrategy === api.core.FeatureLoadingStrategy.all) {
            if (this.loadState === 'all_loading' || this.loadState === 'all_loaded')
                return;
            console.log('Vector:load', this.uid, this.loadingStrategy, 'first load');
            this.loadState = 'all_loading';
            this.loader();

        }
    }


    protected loader() {

        let url = this.props.url;

        url += '?resolution=' + encodeURIComponent(String(this.map.viewState.resolution));
        let bbox = this.map.bbox;

        if (this.loadingStrategy === api.core.FeatureLoadingStrategy.bbox) {
            url += '&bbox=' + encodeURIComponent(bbox.join(','));
        }

        this.map.app.server.dequeueLoad(this.uid);
        let req = this.map.app.server.queueLoad(this.uid, url, '');
        req.then(res => this.olLoaderDone(res));
    }

    protected olLoaderDone(res) {
        this.loadState = '';

        if (!res) {
            console.log('Vector:load', this.uid, this.loadingStrategy, 'EMPTY', res);
            return;
        }
        if (!res.features) {
            console.log('Vector:load', this.uid, this.loadingStrategy, 'ERROR', res);
            return;
        }

        if (this.loadingStrategy === api.core.FeatureLoadingStrategy.all)
            this.loadState = 'all_loaded';


        console.log('Vector:load', this.uid, this.loadingStrategy, 'loaded', res.features.length);

        let oldMap = this.fMap;
        let newMap: FeatureMap = {};

        for (let feature of Object.values(oldMap)) {
            if (feature.isDirty || feature.isNew || feature.isFocused) {
                console.log('Vector:load', this.uid, this.loadingStrategy, 'keep', feature.uid);
                newMap[feature.uid] = feature;
            }
        }

        for (let props of res.features) {
            let model = this.map.app.models.model(props.modelUid);
            let feature = model.featureFromProps(props);
            if (!newMap[feature.uid]) {
                newMap[feature.uid] = feature;
            }
        }

        this.setFeatures(newMap);
        this.map.changed();

    }

    //

    addFeature(feature: types.IFeature) {
        this.addFeatures([feature]);
    }

    addFeatures(features: Array<types.IFeature>) {

        for (let fe of features) {
            if (this.fMap[fe.uid]) {
                this.olRemoveFeature(this.fMap[fe.uid]);
            }
            this.fMap[fe.uid] = fe;
            fe.layer = this;
            this.olAddFeature(fe);
        }
        this.source.changed();
    }

    removeFeature(feature) {
        this.removeFeatures([feature]);
    }

    removeFeatures(features) {
        for (let f of features) {
            this.olRemoveFeature(f);
            delete this.fMap[f.uid];
        }
        this.source.changed();
    }

    clear() {
        this._oLayer.setSource(this.createSource());
        this.fMap = {};
    }


    protected setFeatures(newMap: FeatureMap) {
        let oFeatures = [];
        for (let fe of Object.values(newMap)) {
            if (fe.oFeature)
                oFeatures.push(fe.oFeature);
        }

        this.source.clear();
        this.source.addFeatures(oFeatures);

        this.fMap = newMap;
    }


    protected olAddFeature(f) {
        if (this.source && f.oFeature) {
            this.source.addFeature(f.oFeature);
            f.oFeature.changed();
        }
    }

    protected olRemoveFeature(f) {
        if (this.source && f.oFeature)
            try {
                this.source.removeFeature(f.oFeature);
            } catch (e) {
            }

    }


}
