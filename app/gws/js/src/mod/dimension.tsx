import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as measure from 'gws/map/measure';

import * as sidebar from './sidebar';
import * as toolbar from './toolbar';
import * as toolbox from './toolbox';
import * as storage from './storage';

let {Form, Row, Cell} = gws.ui.Layout;

const STORAGE_CATEGORY = 'Dimension';
const MASTER = 'Shared.Dimension';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as DimensionController;

interface DimensionViewProps extends gws.types.ViewProps {
    controller: DimensionController;
    dimensionSelectedElement: DimensionElement;
    dimensionSelectedText: string;
}

const DimensionStoreKeys = [
    'dimensionSelectedElement',
    'dimensionSelectedText',
    'mapUpdateCount',
];


function dist(x1, y1, x2, y2) {
    return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
}

function slope(a, b) {
    // slope between two points
    let dx = (b[0] - a[0]);
    let dy = (b[1] - a[1]);

    if (dx === 0)
        dx = 0.01;

    return Math.atan(dy / dx);
}

function rotate(p, o, a) {
    // rotate P(oint) about O(rigin) by A(ngle)
    return [
        o[0] + (p[0] - o[0]) * Math.cos(a) - (p[1] - o[1]) * Math.sin(a),
        o[1] + (p[0] - o[0]) * Math.sin(a) + (p[1] - o[1]) * Math.cos(a),
    ]
}

function vpush(ary, x) {
    ary.push(x);
    return ary.length - 1;
}

class DimensionPoint {
    model: DimensionModel;
    index: number;
    coordinate: ol.Coordinate;
    used: number;
    isSelected = false;
    isControl = false;

    constructor(model, coordinate) {
        this.model = model;
        this.coordinate = coordinate;
    }

    get pixel() {
        return this.model.map.oMap.getPixelFromCoordinate(this.coordinate)
    }

}

class DimensionElement {
    model: DimensionModel;
    type: string;
    dimPoints: Array<DimensionPoint>;
    controlPoint: DimensionPoint;
    isSelected: boolean;
    text: string;

    constructor(model, type, dimPoints) {
        this.model = model;
        this.type = type;
        this.dimPoints = dimPoints;
        this.text = '';
    }

    get label() {
        let dim = '';

        if (this.type === 'Line') {
            dim = this.model.formatLength(this.dimPoints[0], this.dimPoints[1])
        }

        return (dim + ' ' + this.text).trim();
    }

    hasPoint(p) {
        return this.dimPoints.indexOf(p) >= 0 || this.controlPoint === p;
    }

    hasSelectedPoints() {
        return this.dimPoints.some(p => p.isSelected) || this.controlPoint.isSelected;
    }

    get coordinates(): Array<ol.Coordinate> {
        return [this.controlPoint].concat(this.dimPoints).map(p => p.coordinate);
    }

    createControlPoint() {
        if (this.type === 'Line') {
            let [x1, y1] = this.dimPoints[0].coordinate;
            let [x2, y2] = this.dimPoints[1].coordinate;

            let cx = x1 + (x2 - x1) / 2;
            let cy = y1 + (y2 - y1) / 2;

            this.controlPoint = this.model.addPoint([cx, cy]);
            this.controlPoint.isControl = true;
        }
    }

    createTag(fragment) {
        if (this.type === 'Line') {
            return this.createLineTag(fragment)
        }
        return [];
    }

    createLineTag(fragment) {

        /*
                                                     label
            [Q1] <-------------------> [Q2] .............. [CP]
             |                           |
             |                           |
             | [h]                       |
             |                           |
             |                           |
             |                           |
            [P1]                        [P2]

            - = modDimensionDimLine
            | = modDimensionDimPlumb
            . = modDimensionDimExt


         */

        let p1 = this.dimPoints[0].coordinate,
            p2 = this.dimPoints[1].coordinate,
            cp = this.controlPoint.coordinate;

        let a = slope(p1, p2);

        let p1r = p1;
        let p2r = rotate(p2, p1, -a);
        let cpr = rotate(cp, p1, -a);

        let h = cpr[1] - p2r[1];

        let q1r = [p1r[0], p1r[1] + h];
        let q2r = [p2r[0], p2r[1] + h];

        let q1 = rotate(q1r, p1, a);
        let q2 = rotate(q2r, p1, a);

        let p1ref = vpush(fragment.points, p1);
        let p2ref = vpush(fragment.points, p2);
        let q1ref = vpush(fragment.points, q1);
        let q2ref = vpush(fragment.points, q2);
        let cpref = vpush(fragment.points, cp);

        fragment.tags.push(['line', {
            class: "modDimensionDimLine",
            'marker-start': 'url(#lineStart)',
            'marker-end': 'url(#lineEnd)',
            x1: ['', q1ref, 0],
            y1: ['', q1ref, 1],
            x2: ['', q2ref, 0],
            y2: ['', q2ref, 1],
        }]);

        fragment.tags.push(['line', {
            class: "modDimensionDimPlumb",
            x1: ['', p1ref, 0],
            y1: ['', p1ref, 1],
            x2: ['', q1ref, 0],
            y2: ['', q1ref, 1],
        }]);

        fragment.tags.push(['line', {
            class: "modDimensionDimPlumb",
            x1: ['', p2ref, 0],
            y1: ['', p2ref, 1],
            x2: ['', q2ref, 0],
            y2: ['', q2ref, 1],
        }]);

        let x1 = q1r[0], x2 = q2r[0];
        let minx, maxx, minref, maxref;

        if (x1 < x2) {
            minx = x1;
            maxx = x2;
            minref = q1ref;
            maxref = q2ref;
        } else {
            minx = x2;
            maxx = x1;
            minref = q2ref;
            maxref = q1ref;
        }

        let anchor = 'middle';

        if (cpr[0] < minx) {
            fragment.tags.push(['line', {
                class: 'modDimensionDimExt',
                x1: ['', minref, 0],
                y1: ['', minref, 1],
                x2: ['', cpref, 0],
                y2: ['', cpref, 1],
            }]);
            anchor = 'start';
        }

        if (cpr[0] > maxx) {
            fragment.tags.push(['line', {
                class: 'modDimensionDimExt',
                x1: ['', cpref, 0],
                y1: ['', cpref, 1],
                x2: ['', maxref, 0],
                y2: ['', maxref, 1],
            }]);
            anchor = 'end';
        }

        let s = this.model.styles['.modDimensionDimLabel'];
        let labelOffset = s ? (s.values.offset_y || 0) : 0;

        fragment.tags.push(['text', {
            class: 'modDimensionDimLabel',
            'text-anchor': anchor,
            x: ['', cpref, 0],
            y: ['', cpref, 1],
            transform: ['rotate', p1ref, p2ref, cpref],
        },
            ['tspan', {}, ''],
            ['tspan', {dy: -labelOffset}, this.label],
        ]);
    }
}

const DimenstionStyles = [
    '.modDimensionDimLine',
    '.modDimensionDimPlumb',
    '.modDimensionDimExt',
    '.modDimensionDimLabel',
    '.modDimensionDimArrow',
    '.modDimensionDimCross',
]

class DimensionModel {
    points: Array<DimensionPoint>;
    elements: Array<DimensionElement>;
    draftCoordinates: Array<ol.Coordinate>;
    draftType: string;
    map: gws.types.IMapManager;
    pixelTolerance = 20;
    index = 0;
    isInteractive: boolean;
    svgDefs: string;
    styles: { [k: string]: gws.types.IStyle };

    constructor(map) {
        this.map = map;
        this.points = [];
        this.elements = [];

        this.styles = {};

        DimenstionStyles.forEach(s => {
            this.styles[s] = this.map.style.get(s);
        });

        let fragment = {points: [], tags: []};
        this.createDefsTag(fragment);
        console.log(fragment);
        this.svgDefs = this.toSvg(fragment.tags[0], []);
    }

    get empty() {
        return this.elements.length === 0 && !this.draftType;
    }

    serialize() {
        this.points.forEach((p, n) => p.index = n);

        return {
            points: this.points.map(p => p.coordinate),
            elements: this.elements.map(e => ({
                type: e.type,
                dimPoints: e.dimPoints.map(p => p.index),
                controlPoint: e.controlPoint.index,
                text: e.text,
            }))

        }
    }

    deserialize(d) {
        this.points = d.points.map(s => new DimensionPoint(this, s));
        this.elements = d.elements.map(s => {
            let e = new DimensionElement(this, s.type, s.dimPoints.map(n => this.points[n]));
            e.text = s.text || '';
            e.controlPoint = this.points[s.controlPoint];
            e.controlPoint.isControl = true;
            return e;
        });
        this.changed();
    }

    changed() {
        this.map.changed();
    }

    setInteractive(interactive) {
        this.isInteractive = interactive;
        this.setDraft(null);
        this.changed();
    }

    selectPoint(point, add = false) {
        if (!add)
            this.unselectAll();
        if (point)
            point.isSelected = true;
        this.changed();
    }

    get selectedPoints() {
        return this.points.filter(p => p.isSelected);
    }

    unselectAll() {
        this.points.forEach(p => p.isSelected = false);
    }

    movePoint(p: DimensionPoint, c: ol.Coordinate) {
        p.coordinate = c;
        this.changed();
    }

    setDraft(type, coordinates = null) {
        this.draftType = type;
        this.draftCoordinates = coordinates;
        this.changed();
    }

    removePoints(points: Array<DimensionPoint>) {
        points.forEach(p => this.elements = this.elements.filter(e => !e.hasPoint(p)));
        this.cleanupPoints();
        this.changed();
    }

    removeElement(element: DimensionElement) {
        this.elements = this.elements.filter(e => e !== element);
        this.cleanupPoints();
        this.changed();
    }

    cleanupPoints() {
        this.points.forEach(p => p.used = 0);
        this.elements.forEach(e => {
            e.dimPoints.forEach(p => p.used++);
            e.controlPoint.used++;
        });
        this.points = this.points.filter(p => p.used > 0);
    }

    addPoint(coordinate) {
        let p = gws.lib.find(this.points, q => q.coordinate[0] === coordinate[0] && q.coordinate[1] === coordinate[1]);
        if (p)
            return p;
        p = new DimensionPoint(this, coordinate);
        this.points.push(p);
        this.changed();
        return p;
    }

    addElement(type, dimPoints) {
        let e = new DimensionElement(this, type, dimPoints);
        e.isSelected = false;
        e.createControlPoint();
        this.elements.push(e)
        this.changed();
    }

    pointAt(pixel) {
        let pts = [];

        this.points.forEach(p => {
            let d = dist(p.pixel[0], p.pixel[1], pixel[0], pixel[1]);
            if (d < this.pixelTolerance)
                pts.push([d, p])
        });

        if (!pts.length)
            return null;

        pts.sort((a, b) => b[0] - a[0]);
        return pts[0][1];
    }

    geoDist(p1, p2) {
        return measure.distance(p1.coordinate, p2.coordinate, this.map.projection, measure.Mode.ELLIPSOID)

    }

    formatLength(p1, p2) {
        let n = this.geoDist(p1, p2);
        if (!n || n < 0.01)
            return '';
        return n.toFixed(2) + ' m';

    }

    toPx(coordinate: ol.Coordinate) {
        try {
            return this.map.oMap.getPixelFromCoordinate(coordinate)
        } catch (e) {
            return [-1, -1];
        }
    }

    toSvg(element, pixels) {
        if (!Array.isArray(element))
            return element;

        let atts = '';

        if (element[1]) {

            atts = gws.lib.entries(element[1]).map(([key, val]) => {

                if (Array.isArray(val)) {
                    switch (val[0]) {
                        case '':
                            val = pixels[val[1]][val[2]];
                            break;
                        case 'rotate':
                            let a = slope(pixels[val[1]], pixels[val[2]])
                            let adeg = a / (Math.PI / 180);
                            let r = pixels[val[3]];
                            val = `rotate(${adeg}, ${r[0]}, ${r[1]})`;
                            break;
                    }
                }

                return `${key}="${val}"`

            }).join(' ');
        }

        let content = element.slice(2).map(el => this.toSvg(el, pixels)).join('');

        if (content) {
            return `<${element[0]} ${atts}>${content}</${element[0]}>`;
        }

        return `<${element[0]} ${atts}/>`;
    }


    drawDraft() {
        let buf = [];
        let r = this.pixelTolerance >> 1;

        if (this.draftType === 'Line') {
            let [x1, y1] = this.toPx(this.draftCoordinates[0]);
            let [x2, y2] = this.toPx(this.draftCoordinates[1]);

            if (x1 >= 0) {
                buf.push(`<circle class="modDimensionDraftPoint" cx="${x1}" cy="${y1}" r="${r}" />`);
            }
            if (x2 >= 0) {
                buf.push(`<circle class="modDimensionDraftPoint" cx="${x2}" cy="${y2}" r="${r}" />`);
            }
            if (x1 >= 0 && x2 >= 0) {
                buf.push(`<line class="modDimensionDraftLine" 
                    x1=${x1}
                    y1=${y1}
                    x2=${x2}
                    y2=${y2}
                />`);
            }
        }
        return buf.join('');

    }

    drawPoint(p) {
        let cls = 'modDimensionPoint';

        if (p.isSelected)
            cls = 'modDimensionSelectedPoint';
        else if (p.isControl)
            cls = 'modDimensionControlPoint';

        let [x, y] = this.toPx(p.coordinate);
        let r = this.pixelTolerance >> 1;

        return `<circle class="${cls}" cx="${x}" cy="${y}" r="${r}" />`;
    }

    createDefsTag(fragment) {
        let buf = [], defs = [];

        let lineStyle = this.styles['.modDimensionDimLine'];

        if (lineStyle && lineStyle.values.marker === 'arrow') {

            let style = this.styles['.modDimensionDimArrow'];

            let w = style ? style.values.width : 0;
            let h = style ? style.values.height : 0;

            w = w || 12;
            h = h || 8;

            defs.push(['marker', {
                id: 'lineStart',
                markerWidth: w,
                markerHeight: h,
                refX: 0,
                refY: h >> 1,
                orient: 'auto',
                markerUnits: 'userSpaceOnUse'
            },
                ['path', {class: 'modDimensionDimArrow', d: `M0,${h >> 1} L${w},0 L${w},${h} Z`}]
            ]);

            defs.push(['marker', {
                id: 'lineEnd',
                markerWidth: w,
                markerHeight: h,
                refX: w,
                refY: h >> 1,
                orient: 'auto',
                markerUnits: 'userSpaceOnUse'
            },
                ['path', {class: 'modDimensionDimArrow', d: `M0,0 L0,${h} L${w},${h >> 1} Z`}]
            ]);

        }

        if (lineStyle && lineStyle.values.marker === 'cross') {

            let style = this.styles['.modDimensionDimCross'];

            let h = style ? style.values.height : 0;

            h = h || 10;

            let m = `
                markerWidth="${h}" 
                markerHeight="${h}" 
                refX="${h >> 1}" 
                refY="${h >> 1}" 
                orient="auto" 
                markerUnits="userSpaceOnUse">
                <path
                    class="modDimensionDimCross"
                    d="M${h},0 L0,${h} "/>
                <path
                    class="modDimensionDimCross"
                    d="M${h >> 1},0 L${h >> 1},${h} "/>
            `;

            buf.push(`<marker id="lineStart" ${m}</marker>`);
            buf.push(`<marker id="lineEnd" ${m}</marker>`);

        }

        fragment.tags.push(['defs', {}, ...defs]);

        // return '<defs>' + buf.join('') + '</defs>';

    }

    draw() {
        if (this.empty)
            return '';

        let mapSize = this.map.oMap.getSize();

        if (!mapSize)
            return '';

        let fragment = {points: [], tags: []};
        this.elements.map(e => e.createTag(fragment));

        let pixels = fragment.points.map(c => this.toPx(c));
        let svg = fragment.tags.map(el => this.toSvg(el, pixels)).join('');
        let points = this.isInteractive ? this.points.map(p => this.drawPoint(p)).join('') : '';
        let draft = this.drawDraft();

        return `<svg width="${mapSize[0]}" height="${mapSize[1]}" version="1.1" xmlns="http://www.w3.org/2000/svg">
            ${this.svgDefs}
            ${points}
            ${svg}
            ${draft}
        </svg>`;
    }

    printItem(): gws.api.PrintItem {
        let fragment = {points: [], tags: [], styles: []};

        this.createDefsTag(fragment);
        this.elements.map(e => e.createTag(fragment));

        let styles = gws.lib.entries(this.styles).map(([name, s]) => ({
            type: gws.api.StyleType.css,
            values: s.values,
            name,
        }));

        return {
            type: 'fragment',
            points: fragment.points,
            tags: fragment.tags,
            styles
        }
    }

}

class DimensionLayer extends gws.map.layer.FeatureLayer {
    master: DimensionController;

    get printItem(): gws.api.PrintItem {
        if (this.master.model.empty)
            return null;
        return this.master.model.printItem()
    }

}

abstract class DimensionTool extends gws.Tool {

    get layer(): DimensionLayer {
        return _master(this).layer;
    }

    get model(): DimensionModel {
        return _master(this).model;
    }

    get toolboxView() {

        let master = _master(this);
        let at = this.getValue('appActiveTool');

        let buttons = [

            <gws.ui.Button
                {...gws.lib.cls('modDimensionModifyButton', at === 'Tool.Dimension.Modify' && 'isActive')}
                tooltip={this.__('modDimensionModifyButton')}
                whenTouched={() => master.startModify()}
            />,
            <gws.ui.Button
                {...gws.lib.cls('modDimensionLineButton', at === 'Tool.Dimension.Line' && 'isActive')}
                tooltip={this.__('modDimensionLineButton')}
                whenTouched={() => master.startDraw('Line')}
            />,
            <gws.ui.Button
                className="modDimensionRemoveButton"
                tooltip={this.__('modDimensionRemoveButton')}
                whenTouched={() => master.removePoints()}
            />,
        ];

        return <toolbox.Content
            title={this.__('modDimension')}
            controller={this}
            buttons={buttons}
        />
    }

    start() {
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Dimension'});
        this.model.setInteractive(true);
    }

    stop() {
        this.model.setInteractive(false);
    }

}

class DimensionLineTool extends DimensionTool {
    lastPoint: DimensionPoint;

    start() {
        super.start();

        this.model.unselectAll();
        this.lastPoint = null;

        // @TODO: join mode
        // let sel = this.model.selectedPoints;
        // this.lastPoint = sel.length > 0 ? sel[0] : null;

        let ixPointer = new ol.interaction.Pointer({
            handleEvent: e => this.handleEvent(e),
        });

        this.map.appendInteractions([
            ixPointer,
            _master(this).snapInteraction(),
        ]);

    }

    stop() {
        super.stop();
        this.finishDrawing();
    }

    handleEvent(e: ol.MapBrowserEvent) {
        if (e.type === 'dblclick') {
            _master(this).startDraw('Line');
            return false;
        }

        if (e.type === 'click') {
            this.addPoint(e);
            return false;
        }

        if (e.type === 'pointermove') {
            let coord = [];

            if (this.lastPoint)
                coord.push(this.lastPoint.coordinate);
            coord.push(e.coordinate);

            this.model.setDraft('Line', coord);
            return false;
        }

        return true;
    }

    addPoint(e) {
        this.model.setDraft(null);

        if (this.lastPoint) {
            let c = this.lastPoint.coordinate;
            if (c[0] === e.coordinate[0] && c[1] === e.coordinate[1]) {
                return;
            }
        }

        let point = this.model.addPoint(e.coordinate);

        if (this.lastPoint) {
            this.model.addElement('Line', [this.lastPoint, point])
        }

        this.lastPoint = point;
    }

    finishDrawing() {
        this.model.setDraft(null);
        this.model.cleanupPoints();
    }

}

class DimensionModifyTool extends DimensionTool {

    point: DimensionPoint;
    ixSnap: ol.interaction.Snap;

    start() {
        super.start();

        let ixPointer = new ol.interaction.Pointer({
            handleEvent: e => this.handleEvent(e),
        });

        this.ixSnap = _master(this).snapInteraction();
        this.map.appendInteractions([ixPointer, this.ixSnap]);
    }

    handleEvent(e: ol.MapBrowserEvent) {
        if (e.type === 'pointerdown') {
            let p = this.model.pointAt(e.pixel);

            if (p) {
                this.model.selectPoint(p);
                this.ixSnap.setActive(!p.isControl);
                this.point = p;
            } else {
                this.model.selectPoint(null);
                this.point = null;
            }
        }

        if (e.type === 'dblclick') {
            if (this.point && this.point.isControl) {
                let element = gws.lib.find(this.model.elements, e => e.controlPoint === this.point);
                _master(this).selectElement(element);
            }
            return false;
        }

        if (e.type === 'pointerdrag') {
            if (this.point) {
                this.model.movePoint(this.point, e.coordinate);
                return false;
            }
        }

        return true;
    }

}

class DimensionController extends gws.Controller {
    uid = MASTER;
    layer: DimensionLayer;
    oOverlay: ol.Overlay;
    setup: gws.api.DimensionProps;
    model: DimensionModel;
    targetUpdateCount = 0;
    snapUpdateCount = 0;

    async init() {

        this.setup = this.app.actionSetup('dimension');
        if (!this.setup)
            return;

        this.model = new DimensionModel(this.map);

        this.model.pixelTolerance = this.setup.pixelTolerance || 10;

        this.layer = this.map.addServiceLayer(new DimensionLayer(this.map, {
            uid: '_dimension',
        }));

        this.layer.master = this;

        this.app.whenChanged('mapRawUpdateCount', () => {
            this.updateOverlay();
            this.checkUpdateSnapInteraction();
        });

        this.setup.layerUids.forEach(uid => {
            this.app.whenChanged('mapLayerUpdateCount_' + uid, () => {
                this.targetUpdateCount++;
            })
        })
    }

    snapFeatures: ol.Collection<ol.Feature>;

    snapInteraction() {
        this.snapFeatures = new ol.Collection<ol.Feature>();
        this.updateSnapInteraction();
        return this.map.snapInteraction({
            features: this.snapFeatures,
            tolerance: this.model.pixelTolerance,
        })
    }

    checkUpdateSnapInteraction() {
        if (this.model.isInteractive && this.targetUpdateCount > this.snapUpdateCount) {
            this.updateSnapInteraction();
            this.snapUpdateCount = this.targetUpdateCount;
        }
    }

    updateSnapInteraction() {
        if (this.snapFeatures && this.setup.layerUids) {
            this.snapFeatures.clear();
            this.setup.layerUids.forEach(uid => {
                let la = (this.map.getLayer(uid) as gws.types.IMapFeatureLayer);
                if (la && la.features)
                    this.snapFeatures.extend(la.features.map(f => f.oFeature));
            });
        }
    }

    updateOverlay() {
        let svg = this.model.draw();

        if (!svg && !this.oOverlay) {
            return;
        }

        if (!this.oOverlay) {
            this.oOverlay = new ol.Overlay({
                element: document.createElement('div'),
                stopEvent: false,
            });

            // old firefoxes have problems unless this is set
            (this.oOverlay.getElement() as HTMLDivElement).style.pointerEvents = 'none';

            this.map.oMap.addOverlay(this.oOverlay);
        }

        let coordinate = this.map.oMap.getCoordinateFromPixel([0, 0]);
        this.oOverlay.setPosition(coordinate);

        this.oOverlay.getElement().innerHTML = svg;
    }

    startDraw(dimType) {
        this.app.startTool('Tool.Dimension.' + dimType)
    }

    startModify() {
        this.app.startTool('Tool.Dimension.Modify')

    }

    removePoints() {
        this.model.removePoints(this.model.selectedPoints)
    }

    selectElement(element?: DimensionElement) {
        this.update({
            dimensionSelectedElement: element,
            dimensionSelectedText: element ? element.text : '',
        });

    }

    clear() {
        this.model.elements = [];
        this.model.points = [];
        this.selectElement(null);
        this.model.changed();
    }

}

class DimensionElementList extends gws.components.list.List<DimensionElement> {

}

class DimensionSidebarView extends gws.View<DimensionViewProps> {
    render() {

        let master = _master(this.props.controller),
            model = master.model;

        let selectedElement = master.getValue('dimensionSelectedElement');


        let zoom = (e: DimensionElement, mode) => {
            let f = master.map.newFeature({
                geometry: new ol.geom.MultiPoint(e.coordinates)
            });
            this.props.controller.update({
                marker: {
                    features: [f],
                    mode
                }
            });
        };

        let focus = (e: DimensionElement) => {
            zoom(e, 'pan fade');
            master.selectElement(e);
        };

        let remove = (e: DimensionElement) => {
            model.removeElement(e)
        };

        let changed = (key, val) => master.update({dimensionSelectedText: val});

        let submit = () => {
            if (selectedElement)
                selectedElement.text = master.getValue('dimensionSelectedText');
            model.changed();
            master.selectElement(null)
        };

        let hasElements = !gws.lib.empty(model.elements);

        let body;

        if (selectedElement) {
            let formData = [{
                name: 'text',
                title: this.__('modDimensionElementText'),
                value: this.props.dimensionSelectedText,
                type: 'text',
                editable: true
            }];

            body = <div className="modAnnotateFeatureDetails">
                <Form>
                    <Row>
                        <Cell flex>
                            <gws.components.sheet.Editor
                                data={formData}
                                whenChanged={changed}
                                whenEntered={submit}
                            />
                        </Cell>
                    </Row>
                    <Row>
                        <Cell flex/>
                        <Cell>
                            <gws.ui.Button
                                className="cmpButtonFormOk"
                                tooltip={this.props.controller.__('modAnnotateSaveButton')}
                                whenTouched={submit}
                            />
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                className="cmpButtonFormCancel"
                                whenTouched={() => {
                                    master.selectElement(null);
                                }}
                            />
                        </Cell>
                    </Row>
                </Form>
            </div>
        } else if (hasElements) {
            body = <DimensionElementList
                controller={this.props.controller}
                items={model.elements}

                isSelected={e => e.hasSelectedPoints()}

                content={e => <gws.ui.Link
                    whenTouched={() => focus(e)}
                    content={e.label}
                />}

                leftButton={e => <gws.components.list.Button
                    className="cmpListZoomListButton"
                    whenTouched={() => zoom(e, 'zoom fade')}/>}

                rightButton={e => <gws.components.list.Button
                    className="modSelectUnselectListButton"
                    whenTouched={() => remove(e)}
                />}
            />
        } else {
            body = <sidebar.EmptyTabBody>
                {this.__('modDimensionNoObjects')}
            </sidebar.EmptyTabBody>
        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modDimensionSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {body}
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                    {storage.auxButtons(master, {
                        category: STORAGE_CATEGORY,
                        hasData: hasElements,
                        getData: name => master.model.serialize(),
                        dataReader: (name, data) => master.model.deserialize(data)
                    })}
                    <sidebar.AuxButton
                        disabled={!hasElements}
                        className="modSelectClearAuxButton"
                        tooltip={this.__('modDimensionClearAuxButton')}
                        whenTouched={() => master.clear()}
                    />
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class DimensionSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modDimensionSidebarIcon';

    get tooltip() {
        return this.__('modDimensionSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(DimensionSidebarView, DimensionStoreKeys)
        );
    }

}

class DimensionToolbarButton extends toolbar.Button {
    iconClass = 'modDimensionToolbarButton';
    tool = 'Tool.Dimension.Line';

    get tooltip() {
        return this.__('modDimensionToolbarButton');
    }

}

export const tags = {
    [MASTER]: DimensionController,

    'Sidebar.Dimension': DimensionSidebar,
    'Toolbar.Dimension': DimensionToolbarButton,
    'Tool.Dimension.Modify': DimensionModifyTool,
    'Tool.Dimension.Line': DimensionLineTool,

};
