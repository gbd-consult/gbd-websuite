import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as measure from 'gws/map/measure';

import * as sidebar from 'gws/elements/sidebar';
import * as toolbar from 'gws/elements/toolbar';
import * as toolbox from 'gws/elements/toolbox';
import * as components from 'gws/components';
import {FormField} from "gws/components/form";
import * as storage from 'gws/elements/storage';


let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Dimension';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as Controller;

interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    dimensionSelectedElement: Element;
    dimensionFormValues: gws.types.Dict;
}

const StoreKeys = [
    'dimensionSelectedElement',
    'dimensionFormValues',
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

class Point {
    model: Model;
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

class Element {
    model: Model;
    type: string;
    dimPoints: Array<Point>;
    controlPoint: Point;
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

    createTag(soup) {
        if (this.type === 'Line') {
            return this.createLineTag(soup)
        }
        return [];
    }

    createLineTag(soup) {

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

            - = dimensionDimLine
            | = dimensionDimPlumb
            . = dimensionDimExt


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

        let p1ref = vpush(soup.points, p1);
        let p2ref = vpush(soup.points, p2);
        let q1ref = vpush(soup.points, q1);
        let q2ref = vpush(soup.points, q2);
        let cpref = vpush(soup.points, cp);

        soup.tags.push(['line', {
            class: "dimensionDimLine",
            'marker-start': 'url(#lineStart)',
            'marker-end': 'url(#lineEnd)',
            x1: ['x', q1ref],
            y1: ['y', q1ref],
            x2: ['x', q2ref],
            y2: ['y', q2ref],
        }]);

        soup.tags.push(['line', {
            class: "dimensionDimPlumb",
            x1: ['x', p1ref],
            y1: ['y', p1ref],
            x2: ['x', q1ref],
            y2: ['y', q1ref],
        }]);

        soup.tags.push(['line', {
            class: "dimensionDimPlumb",
            x1: ['x', p2ref],
            y1: ['y', p2ref],
            x2: ['x', q2ref],
            y2: ['y', q2ref],
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
            soup.tags.push(['line', {
                class: 'dimensionDimExt',
                x1: ['x', minref],
                y1: ['y', minref],
                x2: ['x', cpref],
                y2: ['y', cpref],
            }]);
            anchor = 'start';
        }

        if (cpr[0] > maxx) {
            soup.tags.push(['line', {
                class: 'dimensionDimExt',
                x1: ['x', cpref],
                y1: ['y', cpref],
                x2: ['x', maxref],
                y2: ['y', maxref],
            }]);
            anchor = 'end';
        }

        let s = this.model.styles['.dimensionDimLabel'];
        let labelOffset = s ? (s.values.offset_y || 0) : 0;

        soup.tags.push(['text', {
            class: 'dimensionDimLabel',
            'text-anchor': anchor,
            x: ['x', cpref],
            y: ['y', cpref],
            transform: ['r', p1ref, p2ref, cpref],
        },
            ['tspan', {}, ''],
            ['tspan', {dy: -labelOffset}, this.label],
        ]);
    }
}

const DimenstionStyles = [
    '.dimensionDimLine',
    '.dimensionDimPlumb',
    '.dimensionDimExt',
    '.dimensionDimLabel',
    '.dimensionDimArrow',
    '.dimensionDimCross',
]

class Model {
    points: Array<Point>;
    elements: Array<Element>;
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

        let soup = {points: [], tags: []};
        this.createDefsTag(soup);
        console.log(soup);
        this.svgDefs = this.toSvg(soup.tags[0], []);
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
        this.points = d.points.map(s => new Point(this, s));
        this.elements = d.elements.map(s => {
            let e = new Element(this, s.type, s.dimPoints.map(n => this.points[n]));
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

    movePoint(p: Point, c: ol.Coordinate) {
        p.coordinate = c;
        this.changed();
    }

    setDraft(type, coordinates = null) {
        this.draftType = type;
        this.draftCoordinates = coordinates;
        this.changed();
    }

    removePoints(points: Array<Point>) {
        points.forEach(p => this.elements = this.elements.filter(e => !e.hasPoint(p)));
        this.cleanupPoints();
        this.changed();
    }

    removeElement(element: Element) {
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
        p = new Point(this, coordinate);
        this.points.push(p);
        this.changed();
        return p;
    }

    addElement(type, dimPoints) {
        let e = new Element(this, type, dimPoints);
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
                        case 'x':
                            val = pixels[val[1]][0];
                            break;
                        case 'y':
                            val = pixels[val[1]][1];
                            break;
                        case 'r':
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
                buf.push(`<circle class="dimensionDraftPoint" cx="${x1}" cy="${y1}" r="${r}" />`);
            }
            if (x2 >= 0) {
                buf.push(`<circle class="dimensionDraftPoint" cx="${x2}" cy="${y2}" r="${r}" />`);
            }
            if (x1 >= 0 && x2 >= 0) {
                buf.push(`<line class="dimensionDraftLine"
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
        let cls = 'dimensionPoint';

        if (p.isSelected)
            cls = 'dimensionSelectedPoint';
        else if (p.isControl)
            cls = 'dimensionControlPoint';

        let [x, y] = this.toPx(p.coordinate);
        let r = this.pixelTolerance >> 1;

        return `<circle class="${cls}" cx="${x}" cy="${y}" r="${r}" />`;
    }

    createDefsTag(soup) {
        let buf = [], defs = [];

        let lineStyle = this.styles['.dimensionDimLine'];

        if (lineStyle && lineStyle.values.marker === 'arrow') {

            let style = this.styles['.dimensionDimArrow'];

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
                ['path', {class: 'dimensionDimArrow', d: `M0,${h >> 1} L${w},0 L${w},${h} Z`}]
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
                ['path', {class: 'dimensionDimArrow', d: `M0,0 L0,${h} L${w},${h >> 1} Z`}]
            ]);

        }

        if (lineStyle && lineStyle.values.marker === 'cross') {

            let style = this.styles['.dimensionDimCross'];

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
                    class="dimensionDimCross"
                    d="M${h},0 L0,${h} "/>
                <path
                    class="dimensionDimCross"
                    d="M${h >> 1},0 L${h >> 1},${h} "/>
            `;

            buf.push(`<marker id="lineStart" ${m}</marker>`);
            buf.push(`<marker id="lineEnd" ${m}</marker>`);

        }

        soup.tags.push(['defs', {}, ...defs]);

        // return '<defs>' + buf.join('') + '</defs>';

    }

    draw() {
        if (this.empty)
            return '';

        let mapSize = this.map.oMap.getSize();

        if (!mapSize)
            return '';

        let soup = {points: [], tags: []};
        this.elements.map(e => e.createTag(soup));

        let pixels = soup.points.map(c => this.toPx(c));
        let svg = soup.tags.map(el => this.toSvg(el, pixels)).join('');
        let points = this.isInteractive ? this.points.map(p => this.drawPoint(p)).join('') : '';
        let draft = this.drawDraft();

        return `<svg width="${mapSize[0]}" height="${mapSize[1]}" version="1.1" xmlns="http://www.w3.org/2000/svg">
            ${this.svgDefs}
            ${points}
            ${svg}
            ${draft}
        </svg>`;
    }

    printPlane(): gws.api.core.PrintPlane {
        let soup = {points: [], tags: [], styles: []};

        this.createDefsTag(soup);
        this.elements.map(e => e.createTag(soup));

        let styles = gws.lib.entries(this.styles).map(([name, s]) => ({
            values: s.values,
            name,
        }));

        return {
            type: gws.api.core.PrintPlaneType.soup,
            soupPoints: soup.points,
            soupTags: soup.tags,
        }
    }

}

class Layer extends gws.map.layer.FeatureLayer {
    master: Controller;

    get printPlane(): gws.api.core.PrintPlane {
        if (this.master.model.empty)
            return null;
        return this.master.model.printPlane()
    }

}

abstract class Tool extends gws.Tool {

    get layer(): Layer {
        return _master(this).layer;
    }

    get model(): Model {
        return _master(this).model;
    }

    get toolboxView() {

        let master = _master(this);
        let at = this.getValue('appActiveTool');

        let buttons = [
            <gws.ui.Button
                {...gws.lib.cls('dimensionModifyButton', at === 'Tool.Dimension.Modify' && 'isActive')}
                tooltip={this.__('dimensionModifyButton')}
                whenTouched={() => master.startModify()}
            />,
            <gws.ui.Button
                {...gws.lib.cls('dimensionLineButton', at === 'Tool.Dimension.Line' && 'isActive')}
                tooltip={this.__('dimensionLineButton')}
                whenTouched={() => master.startDraw('Line')}
            />,
            <gws.ui.Button
                className="dimensionRemoveButton"
                tooltip={this.__('dimensionRemoveButton')}
                whenTouched={() => master.removePoints()}
            />,
        ];

        return <toolbox.Content
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

class LineTool extends Tool {
    lastPoint: Point;

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

class ModifyTool extends Tool {

    point: Point;
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


class ElementList extends components.list.List<Element> {

}

class SidebarView extends gws.View<ViewProps> {
    render() {
        if (!this.props.dimensionSelectedElement) {
            return <ListTab {...this.props}/>;
        }
        return <FormTab {...this.props}/>;
    }
}


class ListTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this.props.controller),
            model = cc.model;

        let hasElements = !gws.lib.isEmpty(cc.model.elements);

        let zoom = (e: Element, mode) => {
            let f = cc.app.modelRegistry.defaultModel().featureFromGeometry(
                new ol.geom.MultiPoint(e.coordinates)
            );
            this.props.controller.update({
                marker: {
                    features: [f],
                    mode
                }
            });
        };

        let focus = (e: Element) => {
            zoom(e, 'pan fade');
            cc.selectElement(e);
        };

        let remove = (e: Element) => {
            model.removeElement(e)
        };


        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('dimensionSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {
                    hasElements
                        ? <ElementList
                            controller={this.props.controller}
                            items={model.elements}

                            isSelected={e => e.hasSelectedPoints()}

                            content={e => <gws.ui.Link
                                whenTouched={() => focus(e)}
                                content={e.label}
                            />}

                            leftButton={e => <components.list.Button
                                className="cmpListZoomListButton"
                                whenTouched={() => zoom(e, 'zoom fade')}/>}

                            rightButton={e => <components.list.Button
                                className="dimensionDeleteListButton"
                                whenTouched={() => remove(e)}
                            />}
                        />
                        : <sidebar.EmptyTabBody>
                            {this.__('dimensionNoObjects')}
                        </sidebar.EmptyTabBody>
                }
                    </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                    <storage.AuxButtons
                        controller={cc}
                        actionName="dimensionStorage"
                        hasData={hasElements}
                        getData={() => cc.storageGetData()}
                        loadData={(data) => cc.storageLoadData(data)}
                    />
                    <sidebar.AuxButton
                        className="dimensionClearAuxButton"
                        tooltip={this.__('dimensionClearAuxButton')}
                        whenTouched={() => cc.clear()}
                    />
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}


class FormTab extends gws.View<ViewProps> {
    render() {

        let cc = _master(this.props.controller),
            model = cc.model;

        let fields = []

        fields.push({
            type: "text",
            name: "text",
            title: this.__('dimensionElementText'),
            widgetProps: {
                type: "input",
            }
        })


        let values = this.props.dimensionFormValues || {};

        let form = <Form>
            <Row>
                <Cell flex>
                    <table className="cmpForm">
                        <tbody>
                        {fields.map((f, i) => <FormField
                            key={i}
                            field={f}
                            controller={this.props.controller}
                            feature={null}
                            values={values}
                            widget={cc.createWidget(f, values)}
                        />)}
                        </tbody>
                    </table>
                </Cell>
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        className="cmpButtonFormOk"
                        tooltip={this.props.controller.__('dimensionSaveAuxButton')}
                        whenTouched={() => cc.whenEditOkButtonTouched()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.Button
                        className="cmpButtonFormCancel"
                        whenTouched={() => {
                            cc.selectElement(null);
                        }}
                    />
                </Cell>
            </Row>
        </Form>

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('dimensionSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {form}
            </sidebar.TabBody>

        </sidebar.Tab>
    }
}

class Sidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'dimensionSidebarIcon';

    get tooltip() {
        return this.__('dimensionSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarView, StoreKeys)
        );
    }

}

class ToolbarButton extends toolbar.Button {
    iconClass = 'dimensionToolbarButton';
    tool = 'Tool.Dimension.Line';

    get tooltip() {
        return this.__('dimensionToolbarButton');
    }

}

class Controller extends gws.Controller {
    uid = MASTER;
    layer: Layer;
    oOverlay: ol.Overlay;
    setup: gws.api.plugin.dimension.Props;
    model: Model;
    targetUpdateCount = 0;
    snapUpdateCount = 0;

    async init() {

        this.setup = this.app.actionProps('dimension');
        if (!this.setup)
            return;

        this.model = new Model(this.map);

        this.model.pixelTolerance = this.setup.pixelTolerance || 10;

        this.layer = this.map.addServiceLayer(new Layer(this.map, {
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

        this.updateObject('storageState', {
            dimensionStorage: this.setup.storage ? this.setup.storage.state : null,
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
                let la = (this.map.getLayer(uid) as gws.types.IFeatureLayer);
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

    selectElement(element?: Element) {
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

    createWidget(field: gws.types.IModelField, values: gws.types.Dict): React.ReactElement | null {
        let p = field.widgetProps;

        if (!p)
            return null;

        let tag = 'ModelWidget.' + p.type;
        let controller = (this.app.controllerByTag(tag) || this.app.createControllerFromConfig(this, {tag})) as gws.types.IModelWidget;

        let props: gws.types.Dict = {
            controller,
            field,
            widgetProps: field.widgetProps,
            values,
            whenChanged: val => this.whenWidgetChanged(field, val),
            whenEntered: val => this.whenWidgetEntered(field, val),
        }
        return controller.formView(props)
    }

    whenWidgetChanged(field: gws.types.IModelField, value) {
        let fd = this.getValue('dimensionFormValues') || {};
        this.update({
            dimensionFormValues: {
                ...fd,
                [field.name]: value,
            }
        })
    }

    whenWidgetEntered(field: gws.types.IModelField, value) {
        this.whenEditOkButtonTouched()
    }

    whenEditOkButtonTouched() {
        let selectedElement = this.getValue('dimensionSelectedElement')
        if (selectedElement) {
            let formValues = this.getValue('dimensionFormValues')
            selectedElement.text = formValues['text'] || '';
        }
        this.model.changed();
        this.selectElement(null)


    }


    storageGetData() {
        return this.model.serialize()
    }

    storageLoadData(data) {
        this.model.deserialize(data)
    }

}


gws.registerTags({
    [MASTER]: Controller,

    'Sidebar.Dimension': Sidebar,
    'Toolbar.Dimension': ToolbarButton,
    'Tool.Dimension.Modify': ModifyTool,
    'Tool.Dimension.Line': LineTool,

});
