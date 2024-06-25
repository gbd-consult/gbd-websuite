import * as ol from 'openlayers';
import * as proj4 from 'proj4';

import * as types from '../types';
import * as api from '../core/api';
import * as lib from '../lib';

import * as layer from './layer';
import * as model from './model';

let layerTypes = {
    'box': layer.BoxLayer,
    'group': layer.GroupLayer,
    'leaf': layer.LeafLayer,
    'root': layer.RootLayer,
    'tile': layer.TileLayer,
    'tree': layer.TreeLayer,
    'vector': layer.FeatureLayer,
    'xyz': layer.XYZLayer,
};

import {Feature} from './feature';

import * as interactions from './interactions';
import {IModelRegistry} from "../types";

const POINTER_UPDATE_INTERVAL = 10;
const VIEW_UPDATE_INTERVAL = 1000;

const ANIMATION_DURATION = 600;
const ANIMATION_EASING = ol.easing.easeOut;

const USE_RAW_BITMAPS_FOR_PRINT = false;
const BITMAP_PRINT_DPI_THRESHOLD = 90.7; // matches OGC_SCREEN_PPI on the server

export class MapManager implements types.IMapManager {
    static STOP_WALK = {};

    app: types.IApplication = null;

    crs = '';
    domNode = null;
    oMap: ol.Map = null;
    oView: ol.View = null;
    projection = null;
    root = null;
    resolutions = null;
    extent = null;
    wrapX = false;

    protected connectedToStore = false;
    protected coordinatePrecision = 0;
    protected defaultViewState = null;
    protected updateCount = 0;
    protected intrStack = [];
    protected standardInteractions = {};
    protected props: api.base.map.Props;

    // @TODO this should be 'viewExtent' actually
    get bbox() {
        return this.oView.calculateExtent();
    }

    get viewExtent() {
        return this.oView.calculateExtent();
    }

    get size() {
        return this.oMap.getSize();
    }

    get viewState(): types.MapViewState {
        if (this.oView) {
            let v = this.oView,
                c = v.getCenter(),
                res = v.getResolution(),
                rot = v.getRotation();

            return {
                centerX: c[0],
                centerY: c[1],
                resolution: res,
                scale: lib.res2scale(res),
                rotation: rot,
                angle: lib.rad2deg(rot),
                size: this.oMap.getSize(),
            }
        }
    }

    get style() {
        return this.app.style;
    }

    constructor(app, connectedToStore) {
        this.app = app;
        this.connectedToStore = connectedToStore;
    }

    update(args) {
        if (this.connectedToStore)
            this.app.store.update(args);
    }

    async init(props, appLoc) {
        console.log('MAP_INIT', appLoc);

        this.props = props;
        this.domNode = document.querySelector('.gwsMap');

        this.resolutions = this.props.resolutions;
        this.extent = this.props.extent;
        this.coordinatePrecision = this.props.coordinatePrecision || 0;
        this.wrapX = this.props.wrapX;

        this.defaultViewState = {
            centerX: this.props.center[0],
            centerY: this.props.center[1],
            resolution: this.props.initResolution,
            rotation: 0
        };

        let vs = this.decodeViewState(appLoc['map']);

        this.initOMap(vs || this.defaultViewState);
        this.initLayers();
        this.initEvents();
        this.initInteractions();

        this.setLayerChecked(this.root, true);

        this.changed();

        if (this.connectedToStore) {
            this.app.whenChanged('appLocation', loc => {
                let vs = this.decodeViewState(loc['map']);
                vs = vs || this.defaultViewState;
                console.log('vs from location', vs);
                this.setViewState(vs, true);
            });

            if (this.app.options.showLayers) {
                for (let uid of this.app.options.showLayers) {
                    let la = this.getLayer(uid);
                    if (la) {
                        this.setLayerChecked(la, true);
                    }
                }
            }

            if (this.app.options.hideLayers) {
                for (let uid of this.app.options.hideLayers) {
                    let la = this.getLayer(uid);
                    if (la) {
                        this.setLayerChecked(la, false);
                    }
                }
            }
        }

        return await true;
    }

    changed() {
        if (this.connectedToStore) {
            this.update({
                mapUpdateCount: ++this.updateCount,
                mapRawUpdateCount: ++this.rawUpdateCount,
                mapAttribution: this.visibleAttributions(),
            });
        }
        this.draw();
    }

    forceUpdate() {
        this.walk(this.root, la => la.forceUpdate())
    }


    //

    addLayer(layer, where, parent) {
        this.insertLayer(layer, where, parent);
        this.changed();
    }

    addTopLayer(layer) {
        this.addLayer(layer, 0, this.root);
    }

    addServiceLayer(layer: types.IFeatureLayer) {
        layer.unlisted = true;
        this.addTopLayer(layer);
        return layer;
    }

    removeLayer(layer) {
        if (layer && layer.parent)
            layer.parent.children = layer.parent.children.filter(la => la !== layer);
        this.changed();
    }

    getLayer(uid) {
        let found = null;

        this.walk(this.root, la => {
            if (la.uid === uid) {
                found = la;
                return MapManager.STOP_WALK;
            }
        });

        return found;
    }

    editableLayers() {
        return this.collect(this.root, la => la.editAccess);
    }

    setLayerChecked(layer, on) {
        function exclusiveCheck(la: types.ILayer, parents) {
            if (la.exclusive) {
                // in an exclusive group, if a layer belongs to the currenty checked path
                // it should be checked and others should be unchecked
                if (la.children.some(c => parents.indexOf(c) >= 0)) {
                    la.children.forEach(c => c.checked = parents.indexOf(c) >= 0);
                }
            }
            la.children.forEach(c => exclusiveCheck(c, parents));
        }

        function update(la, hidden) {
            la.hidden = hidden || !la.checked;
            la.children.forEach(c => update(c, la.hidden));
        }

        if (!on) {
            layer.checked = false;
        } else {
            // check all the parent layers
            let parents = [];
            let la = layer;

            while (la) {
                la.checked = true;
                parents.push(la);
                la = la.parent;
            }
            // maintain exclusive groups
            exclusiveCheck(this.root, parents);
        }

        update(this.root, false);
        this.changed();
    }

    setLayerExpanded(layer, on) {
        layer.expanded = on;
        this.changed();
    }

    async selectLayer(layer) {
        if (!layer.description) {
            let res = await this.app.server.mapDescribeLayer({layerUid: layer.uid})
            layer.description = res.content;
        }
        this.walk(this.root, la => la.selected = false);
        layer.selected = true;
        this.update({
            'mapSelectedLayer': layer
        });
        this.changed();
    }

    deselectAllLayers() {
        this.walk(this.root, la => la.selected = false);
        this.update({
            'mapSelectedLayer': null
        });
        this.changed();
    }

    hideAllLayers() {
        this.walk(this.root, la => {
            if (!la.isSystem)
                la.hidden = true
        });
        this.changed();
    }

    setTargetDomNode(node) {
        this.oMap.setTarget(node);
        this.changed();
    }

    //

    constrainScale(scale: number): number {
        let res = lib.clamp(
            lib.scale2res(scale),
            Math.min(...this.resolutions),
            Math.max(...this.resolutions),
        );
        return lib.res2scale(res);
    }

    protected prepareViewState(vs: any) {
        let resolution = vs.resolution;
        if ('scale' in vs) {
            resolution = lib.scale2res(vs.scale);
        }

        let centerX = vs.centerX;
        let centerY = vs.centerY;
        if ('center' in vs) {
            centerX = vs.center[0];
            centerY = vs.center[1];
        }

        let rotation = vs.rotation;
        if ('angle' in vs) {
            rotation = lib.deg2rad(lib.asNumber(vs.angle));
        }

        let p: any = {};

        resolution = Number(resolution);
        centerX = Number(centerX);
        centerY = Number(centerY);
        rotation = Number(rotation);

        if (!Number.isNaN(resolution)) {
            p.resolution = lib.clamp(
                resolution,
                Math.min(...this.resolutions),
                Math.max(...this.resolutions),
            );
        }
        if (!Number.isNaN(centerX) && !Number.isNaN(centerY)) {
            let ext = this.extent;
            p.center = [
                lib.clamp(centerX, Math.min(ext[0], ext[2]), Math.max(ext[0], ext[2])),
                lib.clamp(centerY, Math.min(ext[1], ext[3]), Math.max(ext[1], ext[3])),
            ];
        }
        if (!Number.isNaN(rotation)) {
            p.rotation = this.oView.constrainRotation(
                lib.clamp(rotation, 0, Math.PI * 2)
            );
        }

        return p;
    }

    setViewState(vs: any, animate: boolean) {
        let p = this.prepareViewState(vs);
        this.oView.cancelAnimations();

        // console.log('setViewState, p', p);

        if (animate) {
            return this.oView.animate({
                ...p,
                duration: ANIMATION_DURATION,
                easing: ANIMATION_EASING
            }, () => this.viewChanged('animation'));
        }

        if ('center' in p)
            this.oView.setCenter(p.center);
        if ('resolution' in p)
            this.oView.setResolution(p.resolution);
        if ('rotation' in p)
            this.oView.setRotation(p.rotation);
    }

    setResolution(n, animate) {
        this.setViewState({resolution: n}, animate);
    }

    setScale(n, animate) {
        this.setViewState({scale: n}, animate);
    }

    setNextResolution(delta, animate) {
        let cur = this.viewState.resolution,
            res = (delta > 0)
                ? Math.min(...this.resolutions.filter(r => r > cur))
                : Math.max(...this.resolutions.filter(r => r < cur));

        if (Number.isFinite(res) && res !== cur)
            this.setResolution(res, animate);
    }

    setRotation(n, animate) {
        this.setViewState({rotation: n}, animate);
    }

    setAngle(n, animate) {
        this.setViewState({angle: n}, animate);
    }

    setCenter(c, animate) {
        this.setViewState({center: c}, animate);
    }

    setViewExtent(extent, animate, padding = 0, minScale = 0) {
        if (!extent)
            return;

        // view.fit doesn't have maxResolution, so compute the stuff here
        // @TODO: rotation

        let size = this.oMap.getSize();
        let res = this.oView.getResolutionForExtent(extent, [
            size[0] - (padding || 0) * 2, size[1] - (padding || 0) * 2
        ]);

        let minRes = Math.min(...this.resolutions);
        let maxRes = Math.max(...this.resolutions);

        if (minScale) {
            minRes = Math.min(
                this.viewState.resolution,
                lib.scale2res(minScale));
        }

        this.setViewState({
            center: ol.extent.getCenter(extent),
            resolution: lib.clamp(res, minRes, maxRes)
        }, animate);
    }

    resetViewState(animate) {
        this.setViewState(this.defaultViewState, animate);
    }

    //

    lockInteractions() {
        this.oMap.getInteractions().forEach(ix => ix.setActive(false));
    }

    unlockInteractions() {
        this.oMap.getInteractions().forEach(ix => ix.setActive(true));
    }

    setInteractions(ixs) {
        this.oMap.getInteractions().forEach(intr => intr.setActive(false));
        this.oMap.getInteractions().clear();
        ixs.forEach(intr => {
            if (typeof (intr) === 'string')
                intr = this.standardInteractions[intr];
            this.oMap.addInteraction(intr)
        });
        this.oMap.getInteractions().forEach(intr => intr.setActive(true));
    }

    defaultInteractions = ['DragPan', 'MouseWheelZoom', 'PinchZoom', 'ZoomBox'];

    appendInteractions(ixs) {
        this.setInteractions(this.defaultInteractions.concat(ixs.filter(Boolean)));
    }

    prependInteractions(ixs) {
        this.setInteractions(ixs.filter(Boolean).concat(this.defaultInteractions));
    }

    resetInteractions() {
        this.setInteractions(this.defaultInteractions);
    }

    pushInteractions() {
        let intrs = [];
        this.oMap.getInteractions().forEach(intr => {
            intr.setActive(false);
            intrs.push(intr);
        });
        this.intrStack.push(intrs);
    }

    popInteractions() {
        if (this.intrStack.length) {
            this.setInteractions(this.intrStack.pop());
        }
    }

    drawInteraction(opts) {
        return interactions.draw(this, opts);
    }

    selectInteraction(opts) {
        return interactions.select(this, opts);
    }

    modifyInteraction(opts) {
        return interactions.modify(this, opts);
    }

    snapInteraction(opts) {
        return interactions.snap(this, opts);
    }

    pointerInteraction(opts) {
        return interactions.pointer(this, opts);
    }

    //

    protected initOMap(vs) {
        console.log('initOMap', vs);

        vs = this.oView ? this.viewState : vs;

        this.crs = this.props.crs;

        if (this.crs === 'EPSG:3857') {
            this.projection = ol.proj.get(this.crs);
        } else {
            proj4.defs(this.crs, this.props.crsDef);
            this.projection = ol.proj.get(this.crs);
        }

        if (this.oMap) {
            this.oMap.setLayerGroup(new ol.layer.Group());
            this.oMap = this.oView = null;
        }

        this.oMap = new ol.Map({
            view: new ol.View({
                center: [vs.centerX, vs.centerY],
                resolution: vs.resolution,
                rotation: vs.rotation,
                resolutions: this.resolutions,
                extent: this.extent,
                projection: this.projection,
            }),
            // @TODO: need to pass window.devicePixelRatio around
            pixelRatio: 1,
            controls: [],
            interactions: [],

        });

        this.oView = this.oMap.getView();

        // OL4 doesn't support getConstraints, inject our 'constrainCenter' into a minimized private prop

        this.oView['_gwsMapManager'] = this;

        function constrainCenter(xy) {
            let mm = this['_gwsMapManager']
            let size = mm.oMap.getSize();
            if (!size)
                return xy;

            let res = mm.oView.getResolution();
            let ext = [
                mm.extent[0] + (size[0] / 2) * res,
                mm.extent[1] + (size[1] / 2) * res,
                mm.extent[2] - (size[0] / 2) * res,
                mm.extent[3] - (size[1] / 2) * res,
            ];

            let c = [
                lib.clamp(xy[0], Math.min(ext[0], ext[2]), Math.max(ext[0], ext[2])),
                lib.clamp(xy[1], Math.min(ext[1], ext[3]), Math.max(ext[1], ext[3])),
            ];

            if (mm.wrapX) {
                c[0] = xy[0];

                while (c[0] < mm.extent[0])
                    c[0] += (mm.extent[2] - mm.extent[0])
                while (c[0] > mm.extent[2])
                    c[0] -= (mm.extent[2] - mm.extent[0])
            }
            return c;
        }

        // @TODO this doesn't work properly
        //
        // let proto = Object.getPrototypeOf(this.oView);
        // let cc = proto.constrainCenter;
        // for (let p in proto) {
        //     if (proto[p] === cc) {
        //         proto[p] = constrainCenter;
        //     }
        // }
    }

    protected initLayers() {

        this.root = this.initLayer(this.props.rootLayer);

        this.computeOpacities();
    }

    protected initLayer(props: api.base.layer.Props, parent = null): types.ILayer {
        let cls = layerTypes[props.type];
        if (!cls)
            throw new Error('unknown layer type: ' + props.type);

        let layer = new cls(this, props);

        if (props['layers']) {
            props['layers'].forEach(p => {
                this.initLayer(p, layer)
            });
        }

        if (parent)
            this.insertLayer(layer, -1, parent);

        return layer;
    }

    protected updateViewState() {
        if (this.connectedToStore) {
            let vs = this.viewState;
            this.update({
                mapUpdateCount: ++this.updateCount,
                mapViewState: vs,
                mapCenterX: vs.centerX,
                mapCenterY: vs.centerY,
                mapResolution: vs.resolution,
                mapRotation: vs.rotation,
                mapScale: vs.scale,
                mapAngle: vs.angle,
            });
        }
    }

    protected updateAppLocation() {
        if (this.connectedToStore) {
            let v1 = this.encodeViewState(this.viewState);
            let v2 = this.encodeViewState(this.defaultViewState);
            this.app.updateLocation({map: v1 === v2 ? null : v1});
        }
    }

    formatCoordinate(n) {
        return lib.toFixedMax(n, this.coordinatePrecision)

    }

    protected encodeViewState(vs) {
        let xs = [
            this.formatCoordinate(vs.centerX),
            this.formatCoordinate(vs.centerY),
            lib.res2scale(vs.resolution),
        ];

        if (vs.rotation) {
            xs.push(lib.rad2deg(vs.rotation))
        }

        return xs.join(',');
    }

    protected decodeViewState(h) {
        h = String(h || '').split(',');

        let xs = [
            parseFloat(h[0]),
            parseFloat(h[1]),
            parseInt(h[2])
        ];

        xs.push(h.length === 4 ? parseInt(h[3]) : 0);

        if (xs.some(x => Number.isNaN(x)))
            return null;

        return {
            centerX: xs[0],
            centerY: xs[1],
            resolution: lib.scale2res(xs[2]),
            rotation: lib.deg2rad(xs[3]),
        };
    }

    elevationTimer = null;

    async updatePointerPosition(cc) {
        let x = cc[0] | 0;
        let y = cc[1] | 0;

        this.update({
            mapPointerX: x,
            mapPointerY: y,
        });

        // clearTimeout(this.elevationTimer);
        //
        // this.elevationTimer = setTimeout(async () => {
        //     let pt = new ol.geom.Point([x, y]);
        //     let res = await this.app.server['elevationGetData']({
        //         shape: this.geom2shape(pt),
        //     });
        //     let z;
        //     if (res && res['values']) {
        //         z = Number(res['values'][0])
        //     }
        //     this.update({mapPointerZ: z});
        // }, 500);
    }

    protected externalInteracting = false;
    protected viewChangedFlag = false;

    setInteracting(on) {
        this.externalInteracting = on
    }

    rawUpdateCount = 0;

    protected viewChanged(arg) {
        if (this.connectedToStore) {
            this.update({
                mapRawUpdateCount: ++this.rawUpdateCount,
            })
        }

        this.viewChangedFlag = true
    }

    watchViewChanges() {
        if (this.externalInteracting || this.oView.getInteracting() || this.oView.getAnimating()) {
            return
        }

        if (this.viewChangedFlag) {
            this.updateViewState();
            this.updateAppLocation();
            this.changed();
            this.viewChangedFlag = false;
        }
    }

    protected initEvents() {

        let vs = this.viewState;

        setInterval(() => this.watchViewChanges(), VIEW_UPDATE_INTERVAL)

        this.updatePointerPosition([vs.centerX, vs.centerY]);
        this.viewChanged('init');

        this.oMap.on('pointermove', lib.debounce(
            e => this.updatePointerPosition(e.coordinate),
            POINTER_UPDATE_INTERVAL));

        this.oView.on('change:resolution', () => this.viewChanged('res'));
        this.oView.on('change:center', () => this.viewChanged('center'));
        this.oView.on('change:rotation', () => this.viewChanged('rotation'));
    }

    protected initInteractions() {
        this.standardInteractions = {
            'DragPan': new ol.interaction.DragPan({
                kinetic: new ol.Kinetic(-0.01, 0.1, 100)

            }),
            'PinchZoom': new ol.interaction.PinchZoom({
                constrainResolution: true
            }),
            'MouseWheelZoom': new ol.interaction.MouseWheelZoom({
                constrainResolution: true
            }),
            'ZoomBox': new ol.interaction.DragZoom({
                condition: ol.events.condition.shiftKeyOnly,
                className: 'modZoomBox',
            })
        }
    }

    //


    readFeature(props: api.core.FeatureProps): types.IFeature {
        return this._readFeature(props);

    }

    readFeatures(propsList: Array<api.core.FeatureProps>): Array<types.IFeature> {
        return propsList.map(props => this._readFeature(props));
    }

    _readFeature(props) {
        let model = this.app.modelRegistry.getModel(props.modelUid) || this.app.modelRegistry.defaultModel();
        return model.featureFromProps(props)

    }

    //

    focusedFeature: types.IFeature;

    focusFeature(feature?: types.IFeature) {
        let prev = this.focusedFeature;
        this.focusedFeature = feature;

        if (prev)
            prev.redraw();
        if (feature)
            feature.redraw();

        this.changed();
        this.update({mapFocusedFeature: this.focusedFeature});
    }


    //

    geom2shape(geom) {
        if (!geom)
            return null;

        // NB: this is our geojson extension
        // the server is supposed to deal with this in a reasonable way
        if (geom instanceof ol.geom.Circle) {
            return {
                crs: this.crs,
                geometry: {
                    type: 'Circle',
                    center: (geom as ol.geom.Circle).getCenter(),
                    radius: (geom as ol.geom.Circle).getRadius(),
                }
            }
        }

        let format = new ol.format.GeoJSON({
            defaultDataProjection: this.projection,
            featureProjection: this.projection,
        });
        return {
            crs: this.crs,
            // @TODO: silly
            geometry: JSON.parse(format.writeGeometry(geom))
        }
    }

    shape2geom(shape) {
        if (!shape || !shape.geometry)
            return null;

        // @TODO: shapes are assumed to be in the map projection

        if (shape.geometry.type === 'Circle') {
            return new ol.geom.Circle(
                shape.geometry.center,
                shape.geometry.radius,
            );
        }

        let format = new ol.format.GeoJSON();
        return format.readGeometry(shape.geometry);
    }

    //

    async searchForFeatures(args) {

        let ls = this.searchLayers();
        let params: api.base.search.action.Request = {
            extent: this.bbox,
            keyword: args.keyword || '',
            layerUids: lib.compact(ls.map(la => la.uid)),
            resolution: this.viewState.resolution,
            limit: args.limit || 999,
            tolerance: args.tolerance || '',
        };

        if (args.geometry) {
            params.shapes = [this.geom2shape(args.geometry)]
        }

        let res = await this.app.server.searchFind(params);

        if (res.error) {
            console.log('SEARCH_ERROR', res);
            return [];
        }

        return this.readFeatures(res.features);
    }

    protected searchLayers() {
        // select visible terminal layers, starting with selected layers, if any
        // otherwise with the root
        let roots = this.collect(this.root, la => la.selected);
        if (!roots.length)
            roots = [this.root];

        let layers = [];

        roots.forEach(root => this.walk(root, layer => {
            if (!layer.shouldDraw)
                return MapManager.STOP_WALK;
            if (!layer.isSystem)
                layers.push(layer)
        }));

        return lib.uniq(layers);
    }

    protected async printPlanes(boxRect, dpi): Promise<Array<api.core.PrintPlane>> {
        let _this = this;

        function makeBitmap2(): api.core.PrintPlane {
            let canvas = _this.oMap.getViewport().firstChild as HTMLCanvasElement;

            let rc = canvas.getBoundingClientRect(),
                rb = boxRect,
                cx = rb.left - rc.left,
                cy = rb.top - rc.top;

            let ctx = canvas.getContext('2d'),
                imgData = ctx.getImageData(cx, cy, rb.width, rb.height);

            if (USE_RAW_BITMAPS_FOR_PRINT) {
                return {
                    type: api.core.PrintPlaneType.bitmap,
                    bitmapMode: 'RGBA',
                    bitmapData: imgData.data,
                    bitmapWidth: imgData.width,
                    bitmapHeight: imgData.height,
                };
            }

            let cnv2 = document.createElement('canvas');

            cnv2.width = rb.width;
            cnv2.height = rb.height;

            cnv2.getContext('2d').putImageData(imgData, 0, 0);

            return {
                type: api.core.PrintPlaneType.url,
                url: cnv2.toDataURL()
            };
        }

        async function makeBitmap(layers): Promise<api.core.PrintPlane> {
            let hidden = [];

            _this.walk(_this.root, la => {
                if (!la.hidden && !layers.includes(la)) {
                    hidden.push(la);
                    la.hide();
                }
            });

            let bmp: api.core.PrintPlane;

            await lib.delay(200, () => {
                console.time('creating_bitmap');
                bmp = makeBitmap2();
                console.timeEnd('creating_bitmap');
            });

            hidden.forEach(la => la.show());

            return bmp;
        }

        let mode = 0;

        if (boxRect) {
            if (dpi === 0)
                // draft printing, print everything as a bitmap
                mode = 1;
            else if (dpi <= BITMAP_PRINT_DPI_THRESHOLD)
                // low-res printing, print rasters as bitmaps
                mode = 2;
            else
                // normal printing, print only `isClient` layers as bitmaps
                mode = 3;
        }

        // collect printItems or bitmaps for layers, group sequential bitmaps together

        interface PrintItemOrBitmap {
            pi?: api.core.PrintPlane
            bmpLayers?: Array<types.ILayer>
        }

        let items: Array<PrintItemOrBitmap> = [];

        this.walk(this.root, la => {
            let pi = la.shouldDraw && la.printPlane;

            if (mode === 0 && la.displayMode === 'client') {
                // cannot print `isClient` layers without a boxRect
                return;
            }

            if (pi) {
                let useBitmap = (mode === 1) || (mode === 2 && pi.type === 'raster') || (mode > 0 && la.displayMode === 'client');
                if (useBitmap) {
                    if (items.length > 0 && items[items.length - 1].bmpLayers)
                        items[items.length - 1].bmpLayers.push(la)
                    else
                        items.push({pi: null, bmpLayers: [la]})
                } else {
                    items.push({pi, bmpLayers: null});
                }
            }
        });

        let res: Array<api.core.PrintPlane> = [];

        for (let item of items) {
            if (item.pi) {
                res.push(item.pi)
            } else {
                res.push(await makeBitmap(item.bmpLayers))
            }
        }

        return res;
    }

    async printParams(boxRect, dpi): Promise<api.core.PrintMap> {
        let vs = this.viewState,
            visibleLayers = [];

        this.walk(this.root, la => {
            let pi = la.shouldDraw && la.printPlane;
            if (pi) {
                visibleLayers.push(la.uid);
            }
        });

        return {
            planes: await this.printPlanes(boxRect, dpi),
            rotation: Math.round(lib.rad2deg(vs.rotation)),
            scale: lib.res2scale(vs.resolution),
            center: [vs.centerX, vs.centerY],
            styles: this.style.props,
            visibleLayers,
        }
    }

    //

    protected visibleAttributions() {
        let a = [this.app.project.metadata.attribution];

        this.walk(this.root, (layer: types.ILayer) => {
            if (layer.shouldDraw)
                a.push(layer.attribution);
        });

        return lib.uniq(lib.compact(a));
    }

    protected insertLayer(layer, where, parent) {
        if (parent) {
            layer.parent = parent;
            if (where < 0)
                parent.children.push(layer);
            else
                parent.children.splice(where, 0, layer);
        } else {
            layer.parent = null;
        }
    }

    protected draw() {
        this.walk(this.root, la => la.beforeDraw());

        let layers = [];

        this.walk(this.root, layer => {
            if (!layer.shouldDraw)
                return MapManager.STOP_WALK;
            if (layer.oLayer)
                layers.push(layer)
        });

        this.oMap.setLayerGroup(new ol.layer.Group({
            layers: layers.map(la => la.oLayer).reverse()
        }));

        this.walk(this.root, la => {
            if (layers.indexOf(la) < 0)
                this.app.server.dequeueLoad(la.uid);
        });
    }

    walk(layer, fn: (la: types.ILayer) => any) {
        if (fn(layer) === MapManager.STOP_WALK)
            return;
        layer.children.forEach(la => this.walk(la, fn));
    }

    collect(layer, fn) {
        let ls = [];

        let walk = layer => {
            let r = fn(layer);
            if (r === MapManager.STOP_WALK)
                return;
            if (r)
                ls.push(layer);
            layer.children.forEach(walk);
        };

        walk(layer);
        return ls;
    }

    computeOpacities() {
        function compute(layer, groupOpacity) {
            let val = layer.opacity * groupOpacity;
            layer.setComputedOpacity(val);
            layer.children.forEach(la => compute(la, val));
        }

        compute(this.root, 1);
        this.changed();
    }

}
