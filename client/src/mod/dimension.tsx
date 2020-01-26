import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as measure from 'gws/map/measure';

import * as sidebar from './common/sidebar';
import * as toolbar from './common/toolbar';
import * as toolbox from './common/toolbox';
import * as storage from './common/storage';

let {Form, Row, Cell} = gws.ui.Layout;

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

const STORAGE_CATEGORY = 'dimension.model';

function dist(x1, y1, x2, y2) {
    return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
}

function rotate(x0, y0, x, y, a) {
    return [
        x0 + ((x - x0) * Math.cos(a) - (y - y0) * Math.sin(a)),
        y0 + ((x - x0) * Math.sin(a) + (y - y0) * Math.cos(a)),
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

    draw(fragmentPoints) {
        if (this.type === 'Line') {
            return this.drawLine(fragmentPoints)
        }
        return '';
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

    drawLine(fragmentPoints) {
        let [x1, y1] = this.dimPoints[0].coordinate,
            [x2, y2] = this.dimPoints[1].coordinate,
            [xc, yc] = this.controlPoint.coordinate;

        let dx = (x2 - x1);
        let dy = (y2 - y1);

        if (dx === 0 && dy === 0)
            return '';

        if (dx === 0)
            dx = 0.01;

        let a = Math.atan(dy / dx);

        [xc, yc] = rotate(x1, y1, xc, yc, -a);
        [x2, y2] = rotate(x1, y1, x2, y2, -a);

        let adeg = -a / (Math.PI / 180)

        let h = yc - y1;

        let _bl = vpush(fragmentPoints, [x1, y1]);
        let _tl = vpush(fragmentPoints, [x1, y1 + h]);
        let _br = vpush(fragmentPoints, [x2, y2]);
        let _tr = vpush(fragmentPoints, [x2, y2 + h]);
        let _la = vpush(fragmentPoints, [xc, yc]);

        let buf = '';

        buf += `
            <g transform="rotate(${adeg},<<${_bl} 0 0>>,<<${_bl} 1 0>>)">
        `;

        let lny = -(parseInt(this.model.styles['.modDimensionDimLine'].values['offset-y']) || 0);
        let lay = -(parseInt(this.model.styles['.modDimensionDimLabel'].values['offset-y']) || 0);

        lay += lny;

        buf += `<line
            class="modDimensionDimLine"
            marker-start='url(#lineStart)'
            marker-end='url(#lineEnd)'
            x1="<<${_tl} 0 0>>"
            y1="<<${_tl} 1 ${lny}>>"
            x2="<<${_tr} 0 0>>"
            y2="<<${_tr} 1 ${lny}>>"
        />`;

        buf += `<line
            class="modDimensionDimPlumb"
            x1="<<${_bl} 0 0>>"
            y1="<<${_bl} 1 0>>"
            x2="<<${_tl} 0 0>>"
            y2="<<${_tl} 1 ${lny}>>"
        />`;

        buf += `<line
            class="modDimensionDimPlumb"
            x1="<<${_br} 0 0>>"
            y1="<<${_br} 1 0>>"
            x2="<<${_tr} 0 0>>"
            y2="<<${_tr} 1 ${lny}>>"
        />`;

        let minx = Math.min(x1, x2);
        let maxx = Math.max(x1, x2);
        let anchor = 'middle';

        if (xc < minx) {
            let _ex = vpush(fragmentPoints, [minx, yc]);
            buf += `<line
                class="modDimensionDimLine"
                x1="<<${_la} 0 0>>"
                y1="<<${_la} 1 ${lny}>>"
                x2="<<${_ex} 0 0>>"
                y2="<<${_ex} 1 ${lny}>>"
            />`;
            anchor = 'start';
        }

        if (xc > maxx) {
            let _ex = vpush(fragmentPoints, [maxx, yc]);
            buf += `<line
                class="modDimensionDimLine"
                x1="<<${_la} 0 0>>"
                y1="<<${_la} 1 ${lny}>>"
                x2="<<${_ex} 0 0>>"
                y2="<<${_ex} 1 ${lny}>>"
            />`;
            anchor = 'end';
        }

        buf += `<text 
            text-anchor="${anchor}" 
            class="modDimensionDimLabel"
            x="<<${_la} 0 0>>"
            y="<<${_la} 1 ${lay}>>"
            >
            ${this.label}</text>

        `;

        buf += '</g>';

        return buf;
    }
}

const DimenstionStyles = [
    '.modDimensionDimLine',
    '.modDimensionDimPlumb',
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
    styles: gws.types.Dict;

    constructor(map) {
        this.map = map;
        this.points = [];
        this.elements = [];
        this.svgDefs = this.drawDefs();

        this.styles = {};

        DimenstionStyles.forEach(sel => {
            let s = this.map.style.get(sel);
            this.styles[sel] = {
                text: s ? s.text : '',
                values: s ? s.values : '',
            }
        });

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
        let p = gws.tools.find(this.points, q => q.coordinate[0] === coordinate[0] && q.coordinate[1] === coordinate[1]);
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
        return measure.distance(p1.coordinate, p2.coordinate, this.map.projection, measure.ELLIPSOID)

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

    drawDefs() {
        let buf = [];

        let lineStyle = this.map.style.get('.modDimensionDimLine');

        if (lineStyle && lineStyle.values['mark'] === 'arrow') {

            let style = this.map.style.get('.modDimensionDimArrow');

            let w = style ? parseInt(style.values['width']) : 0;
            let h = style ? parseInt(style.values['height']) : 0;

            w = w || 12;
            h = h || 8;

            buf.push(`
                <marker 
                    id="lineStart" 
                    markerWidth="${w}" 
                    markerHeight="${h}" 
                    refX="0" 
                    refY="${h >> 1}" 
                    orient="auto" 
                    markerUnits="userSpaceOnUse">
                    <path
                        class="modDimensionDimArrow"
                        d="M0,${h >> 1} L${w},0 L${w},${h} Z"/>
                </marker>
                <marker 
                    id="lineEnd" 
                    markerWidth="${w}" 
                    markerHeight="${h}"
                    refX="${w}" 
                    refY="${h >> 1}"
                    orient="auto" 
                    markerUnits="userSpaceOnUse">
                    <path
                        class="modDimensionDimArrow"
                        d="M0,0 L0,${h} L${w},${h >> 1} Z" />
                </marker>
            `);
        }

        if (lineStyle && lineStyle.values['mark'] === 'cross') {

            let style = this.map.style.get('.modDimensionDimCross');

            let h = style ? parseInt(style.values['height']) : 0;

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

        return '<defs>' + buf.join('') + '</defs>';

    }

    draw() {
        if (this.empty)
            return '';

        let mapSize = this.map.oMap.getSize();

        if (!mapSize)
            return '';

        let frag = this.fragment();

        let xy = frag.points.map(c => this.toPx(c));

        frag.svg = frag.svg.replace(/<<(\S+) (\S+) (\S+)>>/g, ($0, $1, $2, $3) => {
            let px = xy[Number($1)]
            return String(px[Number($2)] + Number($3))
        });

        let points = this.isInteractive ? this.points.map(p => this.drawPoint(p)).join('') : '';

        let draft = this.drawDraft();

        return `<svg width="${mapSize[0]}" height="${mapSize[1]}" version="1.1" xmlns="http://www.w3.org/2000/svg">
            ${this.svgDefs}
            ${points}
            ${frag.svg}
            ${draft}
        </svg>`;
    }

    fragment(): gws.api.SvgFragment {
        let fragmentPoints = [];
        let elements = this.elements.map(e => e.draw(fragmentPoints)).join('');

        return {
            points: fragmentPoints,
            svg: `${this.svgDefs}${elements}`
        }
    }

    printFragment() {
        let css = gws.tools.entries(this.styles).map(([sel, s]) =>
            sel + ' {\n' + s['source'] + '\n}'
        ).join('\n');

        let frag = this.fragment();
        frag.svg = `
            <style>${css}</style>${frag.svg}
        `;

        return frag;
    }

}

class DimensionLayer extends gws.map.layer.FeatureLayer {
    master: DimensionController;

    get printItem(): gws.api.PrintItem {
        return {
            type: 'fragment',
            printAsVector: true,
            fragment: this.master.model.printFragment(),
        };
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
                {...gws.tools.cls('modDimensionModifyButton', at === 'Tool.Dimension.Modify' && 'isActive')}
                tooltip={this.__('modDimensionModifyButton')}
                whenTouched={() => master.startModify()}
            />,
            <gws.ui.Button
                {...gws.tools.cls('modDimensionLineButton', at === 'Tool.Dimension.Line' && 'isActive')}
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
                let element = gws.tools.find(this.model.elements, e => e.controlPoint === this.point);
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
                if (la)
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

        let hasElements = !gws.tools.empty(model.elements);

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
                    <storage.ReadAuxButton
                        controller={this.props.controller}
                        category={STORAGE_CATEGORY}
                        whenDone={data => master.model.deserialize(data)}
                    />
                    {hasElements && <storage.WriteAuxButton
                        controller={this.props.controller}
                        category={STORAGE_CATEGORY}
                        data={master.model.serialize()}
                    />}
                    {hasElements && <sidebar.AuxButton
                        className="modSelectClearAuxButton"
                        tooltip={this.__('modDimensionClearAuxButton')}
                        whenTouched={() => master.clear()}
                    />}
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
