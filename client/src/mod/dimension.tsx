import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as sidebar from './common/sidebar';
import * as toolbar from './common/toolbar';
import * as toolbox from './common/toolbox';
import * as draw from './common/draw';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Dimension';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as DimensionController;

class DimensionFeature extends gws.map.Feature {
    isSelected: boolean;

    get dimType() {
        return this.props.attributes.dimType;
    }

    get title() {
        let coords = (this.geometry as ol.geom.MultiPoint).getCoordinates();
        return this.map.formatCoordinate(coords[0][0]) + ', ' + this.map.formatCoordinate(coords[0][1])
    }

    toPx(coord) {
        try {
            let [x, y] = this.map.oMap.getPixelFromCoordinate(coord);
            return [x | 0, y | 0];
        } catch (e) {
            return [0, 0];
        }
    }

    asSVG() {
        let coords = (this.geometry as ol.geom.LineString).getCoordinates();
        let buf = coords.map(c => this.anchor(c));

        switch (this.props.attributes.dimType) {
            case 'Line':
                for (let n = 0; n < coords.length - 1; n++)
                    buf.push(this.segment(coords[n], coords[n + 1]));
                break;
            case 'Arc':
                for (let n = 0; n < coords.length - 2; n += 2)
                    buf.push(this.arc(coords[n], coords[n + 1], coords[n + 2]))
                break;
            case 'Circle':
                for (let n = 0; n < coords.length - 1; n += 2)
                    buf.push(this.circle(coords[n], coords[n + 1]))
                break;

        }

        return buf.map(x => x.trim()).filter(Boolean).join('\n');
    }

    segment(p1, p2) {
        let [x1, y1] = this.toPx(p1),
            [x2, y2] = this.toPx(p2);

        let mx = Math.min(x1, x2) + (Math.abs(x1 - x2) >> 1);
        let my = Math.min(y1, y2) + (Math.abs(y1 - y2) >> 1);

        let phi = gws.tools.rad2deg(Math.atan((y2 - y1) / (x2 - x1)));

        let length = formatMeters(this.geoDist(p1, p2));

        let buf = `<line                 
            class="modDimensionDimLine"
            x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" 
        />`;

        if (length && !isNaN(phi) && dist(x1, y1, x2, y2) > 50) {
            buf += `<text 
                class="modDimensionDimLabel"
                x="${mx}" y="${my - 5}" 
                text-anchor="middle" 
                transform="rotate(${phi},${mx},${my})">
                ${length}
            </text>`;
        }

        return buf;
    }

    arc(p1, p2, p3) {
        // maths based on https://stackoverflow.com/a/43825818/989121

        let [x1, y1] = this.toPx(p1);
        let [x2, y2] = this.toPx(p2);
        let [x3, y3] = this.toPx(p3);

        let a = dist(x1, y1, x3, y3);
        let b = dist(x1, y1, x2, y2);
        let c = dist(x3, y3, x2, y2);

        // arc angle - cosine theorem
        // |12|^2 + |23|^2 - |12|^2 =  2cos(phi)

        let cos = (b * b + c * c - a * a) / (2 * b * c);

        if (Math.abs(cos) >= 1)
            return '';

        // arc radius - sine theorem
        // a / 2sin(phi) = r

        let phi = Math.acos(cos);
        let r = (a / (2 * Math.sin(phi))) | 0;

        // large arc flag = 1 if phi < 90
        let large = phi > (Math.PI / 2) ? 0 : 1;

        // sweep flag = 1 if p2 if below the line p1-p3
        let sweep = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) > 0 ? 0 : 1;

        let p = `M${x1} ${y1} A ${r} ${r} 0 ${large} ${sweep} ${x3} ${y3}`;

        let color = this.isSelected ? '#ff0000' : '#000000';

        let radius = formatMeters(this.geoDist(p1, p3) / (2 * Math.sin(phi)));

        return `
            <path
                class="modDimensionDimLine"
                d="${p}" 
                stroke="${color}" 
            />
            <path
                class="modDimensionDimGuide"
                d="M${x2} ${y2} L${x2 + 50} ${y2 - 50} L${x2 + 100} ${y2 - 50}"
            />
            <text
                class="modDimensionDimLabel"
                x="${x2 + 50}"
                y="${y2 - 50 - 5}"
                >R ${radius}</text>


        `;
    }

    circle(p1, p2) {
        let [x1, y1] = this.toPx(p1);
        let [x2, y2] = this.toPx(p2);

        let r = Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2)) | 0;

        let radius = formatMeters(this.geoDist(p1, p2));

        return `
            <circle 
                class="modDimensionDimLine"
                cx="${x1}" cy="${y1}" r="${r}"
            />
            <path
                class="modDimensionDimGuide"
                d="M${x1} ${y1} L${x1 + 50} ${y1 - 50} L${x1 + 100} ${y1 - 50}"
            />
            <text
                class="modDimensionDimLabel"
                x="${x1 + 50}"
                y="${y1 - 50 - 5}"
                >R ${radius}m</text>
            
        `;

    }

    anchor(p) {
        let [x, y] = this.toPx(p);
        let size = 4;

        return `
            <line 
                class="modDimensionDimMark"
                x1="${x - size}" 
                y1="${y - size}" 
                x2="${x + size}" 
                y2="${y + size}"
            />
            <line 
                class="modDimensionDimMark"
                x1="${x - size}" 
                y1="${y + size}" 
                x2="${x + size}" 
                y2="${y - size}"
            />
        `;
    }

    geoDist(p1, p2) {
        return ol.Sphere.getLength(
            new ol.geom.LineString([p1, p2]),
            {projection: this.map.projection});

    }

}

function dist(x1, y1, x2, y2) {
    return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2)) | 0;
}

function formatMeters(n) {
    if (!n || n < 0.01)
        return '';
    return n.toFixed(2) + ' m';

}

class DimensionLayer extends gws.map.layer.FeatureLayer {

    get selectedFeature(): DimensionFeature {
        let fs = this.features.filter(f => f.isSelected);
        return fs.length ? fs[0] : null;

    }

    asSVG() {
        let mapSize = this.map.oMap.getSize();

        if (!mapSize)
            return '';

        let w = mapSize[0];
        let h = mapSize[1];

        let elems = this.features.map(el => el.asSVG())

        if (elems.length === 0)
            return '';

        let svg = `
            <svg width="${w}" height="${h}" version="1.1" xmlns="http://www.w3.org/2000/svg">
        `;

        svg += gws.tools.compact(elems).join('\n');
        svg += '</svg>'

        return svg;
    }

    selectFeature(f: DimensionFeature) {
        f.isSelected = true;
        this.source.changed();
    }

    unselectAll() {
        this.features.forEach(f => f.isSelected = false);
        this.source.changed();
    }

}

abstract class DimensionTool extends gws.Tool {

    get layer(): DimensionLayer {
        return _master(this).layer;
    }

    get toolboxView() {

        let master = _master(this);
        let at = this.getValue('appActiveTool');

        let buttons = [

            <gws.ui.IconButton
                {...gws.tools.cls('modDimensionModifyButton', at === 'Tool.Dimension.Modify' && 'isActive')}
                tooltip={this.__('modDimensionModifyButton')}
                whenTouched={() => master.startEdit()}
            />,
            <gws.ui.IconButton
                {...gws.tools.cls('modDimensionLineButton', at === 'Tool.Dimension.Line' && 'isActive')}
                tooltip={this.__('modDimensionLineButton')}
                whenTouched={() => master.startDraw('Line')}
            />,
            <gws.ui.IconButton
                {...gws.tools.cls('modDimensionArcButton', at === 'Tool.Dimension.Arc' && 'isActive')}
                tooltip={this.__('modDimensionArcButton')}
                whenTouched={() => master.startDraw('Arc')}
            />,
            <gws.ui.IconButton
                {...gws.tools.cls('modDimensionCircleButton', at === 'Tool.Dimension.Circle' && 'isActive')}
                tooltip={this.__('modDimensionCircleButton')}
                whenTouched={() => master.startDraw('Circle')}
            />,
            <gws.ui.IconButton
                className="modDimensionRemoveButton"
                tooltip={this.__('modDimensionRemoveButton')}
                whenTouched={() => master.removeSelected()}
            />,
        ];

        /*
            <gws.ui.IconButton
                className="modDimensionDrawCommitButton"
                tooltip={this.__('modDimensionDrawCommitButton')}
                whenTouched={() => master.app.call('drawCommit')}
            />,
            <gws.ui.IconButton
                className="modDimensionDrawEndButton"
                tooltip={this.__('modDimensionCancelButton')}
                whenTouched={() => master.app.call('drawCancel')}
            />,

         */

        return <toolbox.Content
            controller={this}
            buttons={buttons}
        />
    }

}

class DimensionDrawTool extends DimensionTool {
    state: string;
    dimType: string;
    ixDraw: any;
    draftFeature: DimensionFeature;

    async init() {
        await super.init();
        this.app.whenCalled('drawCommit', () => {
            if (this.state === 'draw') {
                this.ixDraw.finishDrawing()
            }
        });
        this.app.whenCalled('drawCancel', () => {
            if (this.state === 'draw') {
                this.state = 'cancel';
                this.ixDraw.finishDrawing()
            }
        });
    }

    drawStarted(oFeatures) {
        this.layer.unselectAll();

        this.draftFeature = new DimensionFeature(this.map, {
            props: {attributes: {dimType: this.dimType}},
            oFeature: oFeatures[0],
        });

        this.draftFeature.oFeature.on('change', () => {
            let geom = this.draftFeature.geometry as ol.geom.LineString;
            let coords = geom.getCoordinates();
            if (this.dimType === 'Circle' && coords.length > 2)
                geom.setCoordinates([coords[0], coords[coords.length - 1]]);
            if (this.dimType === 'Arc' && coords.length > 3)
                geom.setCoordinates([coords[0], coords[1], coords[coords.length - 1]]);
        });

        _master(this).layer.addFeature(this.draftFeature)

        this.state = 'draw';
    }

    drawEnded() {
        let master = _master(this);
        let coords = (this.draftFeature.geometry as ol.geom.LineString).getCoordinates();

        _master(this).layer.removeFeature(this.draftFeature)

        if (this.state === 'draw') {
            master.addFeature(this.dimType, coords, null);
            _master(this).whenDrawEnded()

        } else {
            _master(this).whenDrawCanceled()
        }

        this.state = '';
    }

    start() {
        this.ixDraw = this.map.drawInteraction({
            shapeType: 'LineString',
            style: this.map.getStyleFromSelector('.modDimensionFeature'),
            whenStarted: oFeatures => this.drawStarted(oFeatures),
            whenEnded: () => this.drawEnded(),
        });

        this.map.appendInteractions([
            this.ixDraw,
            _master(this).snapInteraction()
        ]);
    }

}

class DimensionLineTool extends DimensionDrawTool {
    dimType = 'Line'
}

class DimensionArcTool extends DimensionDrawTool {
    dimType = 'Arc'
}

class DimensionCircleTool extends DimensionDrawTool {
    dimType = 'Circle'
}

class DimensionModifyTool extends DimensionTool {
    oFeatureCollection: ol.Collection<ol.Feature>;

    select() {
        this.oFeatureCollection.clear();

        let sel = this.layer.selectedFeature;
        if (sel) {
            this.oFeatureCollection.push(sel.oFeature);
        }
    }

    start() {
        this.oFeatureCollection = new ol.Collection<ol.Feature>();

        this.select();

        let ixSelect = this.map.selectInteraction({
            layer: this.layer,
            style: this.map.getStyleFromSelector('.modDimensionSelectedFeature'),
            whenSelected: oFeatures => {
                this.layer.unselectAll();

                if (oFeatures[0] && oFeatures[0]['_gwsFeature']) {
                    this.layer.selectFeature(oFeatures[0]['_gwsFeature']);
                    this.select();
                    //this.whenSelected(f);
                } else {
                    this.oFeatureCollection.clear();
                    //this.whenUnselected();
                }
            }
        });

        let ixModify = this.map.modifyInteraction({
            features: this.oFeatureCollection,
            style: this.map.getStyleFromSelector('.modDimensionSelectedFeature'),
            allowDelete: () => this.layer.selectedFeature.dimType === 'Line',
            allowInsert: () => this.layer.selectedFeature.dimType === 'Line',
            whenEnded: oFeatures => {
                //_master(this).commitEdit()
            }
        });

        this.map.appendInteractions([
            ixSelect, ixModify, _master(this).snapInteraction()
        ]);

    }

    whenSelected(f) {
        //_master(this).selectFeature(f, false);
    }

    whenUnselected() {
        //_master(this).unselectFeature()
    }

    stop() {
        super.stop()
    }
}

class DimensionController extends gws.Controller {
    uid = MASTER;
    layer: DimensionLayer;
    oOverlay: ol.Overlay;
    options: gws.api.DimensionOptionsResponse;

    async init() {
        this.layer = this.map.addServiceLayer(new DimensionLayer(this.map, {
            uid: '_dimensionPoint',
            style: this.map.getStyleFromSelector('.modDimensionFeature')
        }));
        this.oOverlay = new ol.Overlay({
            element: document.createElement('div'),
            stopEvent: false,
        });

        this.map.oMap.addOverlay(this.oOverlay);

        this.app.whenChanged('mapRawUpdateCount', () => this.updateOverlay())

        this.layer.oLayer.getSource().on('change', () => {
            this.updateOverlay()
        });

        await this.loadFileNames()

        this.options = await this.app.server.dimensionOptions({
            projectUid: this.app.project.uid,
        });
    }

    async loadFileNames() {
        let res = await this.app.server.dimensionFileList({
            projectUid: this.app.project.uid,
        });

        this.update({
            dimensionFileNames: res.fileNames,
        });

    }

    snapInteraction() {
        let layer = this.map.getLayer(this.options.layerUid) || this.layer;

        return this.map.snapInteraction({
            layer: layer as gws.types.IMapFeatureLayer,
            tolerance: this.options.tolerance || 10,
        })
    }

    updateOverlay() {
        let coord = this.map.oMap.getCoordinateFromPixel([0, 0])
        this.oOverlay.setPosition(coord);
        this.oOverlay.getElement().innerHTML = this.layer.asSVG();
    }

    startDraw(dimType) {
        this.app.startTool('Tool.Dimension.' + dimType)
    }

    startEdit() {
        this.app.startTool('Tool.Dimension.Modify')

    }

    whenDrawEnded() {
        this.startEdit();
    }

    whenDrawCanceled() {
        this.startEdit();
    }

    removeSelected() {
        let f = this.layer.selectedFeature;
        if (f) {
            this.removeFeature(f);
        }
    }

    removeFeature(f) {
        this.layer.removeFeature(f);
        this.startEdit();
        this.update({
            'dimensionFeatures': this.layer.features,
        })
    }

    focusFeature(f) {
        this.layer.unselectAll();
        this.layer.selectFeature(f);
        this.update({
            marker: {
                features: [f],
                mode: 'zoom'
            }
        });
        this.startEdit();
    }

    addFeature(dimType, coords, uid) {
        uid = uid || 'd' + Number(new Date());

        let geometry = (dimType === 'Line') ? new ol.geom.LineString(coords) : new ol.geom.MultiPoint(coords);

        this.layer.addFeature(new DimensionFeature(this.map, {
            props: {uid, attributes: {dimType}},
            geometry,
        }));

        this.update({
            'dimensionFeatures': this.layer.features,
        });

        this.updateOverlay();

    }

    async save() {
        let res = await this.app.server.dimensionFileWrite({
            projectUid: this.app.project.uid,
            fileName: this.getValue('dimensionFileName'),
            features: this.layer.features.map(f => ({
                ...f.props,
                shape: f.shape,
            }))
        });

        await this.loadFileNames();

        this.update({
            dimensionDialogMode: null,
        });
    }

    async load() {
        let res = await this.app.server.dimensionFileRead({
            projectUid: this.app.project.uid,
            fileName: this.getValue('dimensionFileName'),
        });

        if (res.features) {
            this.layer.clear();
            this.map.readFeatures(res.features).forEach(f => this.addFeature(
                f.props.attributes.dimType,
                (f.geometry as ol.geom.LineString).getCoordinates(),
                f.uid
            ));
        }

        this.update({
            dimensionDialogMode: null,
        });

        this.updateOverlay()
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(DimensionDialog, DimensionStoreKeys));
    }

}

interface DimensionViewProps extends gws.types.ViewProps {
    controller: DimensionController;
    dimensionFeatures: string;
    dimensionDialogMode: string;
    dimensionFileName: string;
    dimensionFileNames: Array<string>;
}

const DimensionStoreKeys = [
    'dimensionFeatures',
    'dimensionDialogMode',
    'dimensionFileName',
    'dimensionFileNames',
    'mapUpdateCount',
];

class DimensionSidebarView extends gws.View<DimensionViewProps> {
    render() {

        let master = _master(this.props.controller);

        let features = master.layer.features;
        let hasFeatures = !gws.tools.empty(features);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modDimensionSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {hasFeatures
                    ? <gws.components.feature.List
                        controller={this.props.controller}
                        features={features}

                        content={(f) => <gws.ui.Link
                            whenTouched={() => master.focusFeature(f)}
                            content={(f as DimensionFeature).title}
                        />}

                        withZoom

                        rightButton={f => <gws.components.list.Button
                            className="modSelectUnselectListButton"
                            whenTouched={() => master.removeFeature(f)}
                        />}
                    />
                    : <sidebar.EmptyTabBody>
                        {this.__('modDimensionNoObjects')}

                    </sidebar.EmptyTabBody>
                }
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                    <sidebar.AuxButton
                        className="modSelectSaveAuxButton"
                        tooltip={this.__('modDimensionSaveAuxButton')}
                        whenTouched={() => master.update({dimensionDialogMode: 'save'})}
                    />
                    <sidebar.AuxButton
                        className="modSelectLoadAuxButton"
                        tooltip={this.__('modDimensionLoadAuxButton')}
                        whenTouched={() => master.update({dimensionDialogMode: 'load'})}
                    />
                    <sidebar.AuxButton
                        className="modSelectClearAuxButton"
                        tooltip={this.__('modDimensionClearAuxButton')}
                        whenTouched={() => master.layer.clear()}
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
    tool = 'Tool.Dimension.Modify';

    get tooltip() {
        return this.__('modDimensionToolbarButton');
    }

}

class DimensionDialog extends gws.View<DimensionViewProps> {

    render() {
        let mode = this.props.dimensionDialogMode;

        if (!mode)
            return null;

        let close = () => this.props.controller.update({dimensionDialogMode: null});
        let update = v => this.props.controller.update({dimensionFileName: v});

        let title, submit, control;

        if (mode === 'save') {
            title = this.__('modDimensionSaveDialogTitle');
            submit = () => this.props.controller.save();
            control = <gws.ui.TextInput
                value={this.props.dimensionFileName}
                whenChanged={update}
                whenEntered={submit}
            />;
        }

        if (mode === 'load') {
            title = this.__('modDimensionLoadDialogTitle');
            submit = () => this.props.controller.load();
            control = <gws.ui.Select
                value={this.props.dimensionFileName}
                items={this.props.dimensionFileNames.map(s => ({
                    text: s,
                    value: s

                }))}
                whenChanged={update}
            />;
        }

        return <gws.ui.Dialog className="modSelectDialog" title={title} whenClosed={close}>
            <Form>
                <Row>
                    <Cell flex>{control}</Cell>
                </Row>
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton className="cmpButtonFormOk" whenTouched={submit}/>
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton className="cmpButtonFormCancel" whenTouched={close}/>
                    </Cell>
                </Row>
            </Form>
        </gws.ui.Dialog>;
    }
}

export const tags = {
    [MASTER]: DimensionController,

    'Sidebar.Dimension': DimensionSidebar,
    'Toolbar.Dimension': DimensionToolbarButton,
    'Tool.Dimension.Modify': DimensionModifyTool,
    'Tool.Dimension.Line': DimensionLineTool,
    'Tool.Dimension.Arc': DimensionArcTool,
    'Tool.Dimension.Circle': DimensionCircleTool,

};
