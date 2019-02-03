import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as sidebar from './common/sidebar';
import * as toolbar from './common/toolbar';
import * as modify from './common/modify';
import * as draw from './common/draw';

const MASTER = 'Shared.Annotate';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as AnnotateController;

let {Form, Row, Cell} = gws.ui.Layout;

const defaultLabelTemplates = {
    Point: '{x}, {y}',
    Line: '{len}',
    Polygon: '{area}',
    Circle: '{radius}',
    Box: '{w} x {h}',
};

interface AnnotateFormData {
    x?: string
    y?: string
    w?: string
    h?: string
    radius?: string
    labelTemplate: string
}

interface AnnotateFeatureArgs extends gws.types.IMapFeatureArgs {
    labelTemplate: string;
    selectedStyle: gws.types.IMapStyle;
    shapeType: string;
}

interface AnnotateViewProps extends gws.types.ViewProps {
    controller: AnnotateController;
    mapUpdateCount: number;
    annotateSelectedFeature: AnnotateFeature;
    annotateFormData: AnnotateFormData;
    appActiveTool: string;
};

const AnnotateStoreKeys = [
    'mapUpdateCount',
    'annotateSelectedFeature',
    'annotateFormData',
    'appActiveTool',
];

class AnnotateFeature extends gws.map.Feature {
    master: AnnotateController;
    labelTemplate: string;
    selected: boolean;
    selectedStyle: gws.types.IMapStyle;
    shapeType: string;
    cc: any;

    get dimensions() {
        return computeDimensions(this.shapeType, this.geometry, this.map.projection);
    }

    get formData(): AnnotateFormData {
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

    constructor(master: AnnotateController, args: AnnotateFeatureArgs) {
        super(master.map, args);
        this.master = master;
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
        let dims = this.dimensions;
        this.setLabel(formatTemplate(this, this.labelTemplate, dims));
        this.master.featureUpdated(this);
    }

    updateFromForm(ff: AnnotateFormData) {
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

function formatCoordinate(feature: AnnotateFeature, n) {
    return feature.map.formatCoordinate(Number(n) || 0);

}

function formatTemplate(feature: AnnotateFeature, text, dims) {

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

class AnnotateLayer extends gws.map.layer.FeatureLayer {
}

class AnnotateDrawTool extends draw.Tool {
    drawFeature: AnnotateFeature;

    whenStarted(shapeType, oFeature) {
        this.drawFeature = _master(this).newFeature(shapeType, oFeature);
    }

    whenEnded() {
        _master(this).addAndFocus(this.drawFeature);
    }

    whenCancelled() {
        _master(this).app.startTool('Tool.Annotate.Modify')
    }

}

class AnnotateModifyTool extends modify.Tool {
    get layer() {
        return _master(this).layer;
    }

    start() {
        super.start();
        let f = this.getValue('annotateSelectedFeature');
        if (f)
            this.selectFeature(f);

    }

    whenSelected(f) {
        _master(this).selectFeature(f, false);
    }

    whenUnselected() {
        _master(this).unselectFeature()
    }
}

class AnnotateFeatureDetailsToolbar extends gws.View<AnnotateViewProps> {
    render() {
        return <Row>
            <Cell flex/>
        </Row>;

    }
}

class AnnotateFeatureDetailsForm extends gws.View<AnnotateViewProps> {

    formAttributes(f: AnnotateFeature, ff: AnnotateFormData): Array<gws.components.sheet.Attribute> {
        switch (f.shapeType) {
            case 'Point':
                return [
                    {
                        name: 'x',
                        title: this.__('modAnnotateX'),
                        value: ff.x,
                        editable: true
                    },
                    {
                        name: 'y',
                        title: this.__('modAnnotateY'),
                        value: ff.y,
                        editable: true
                    }];

            case 'Box':
                return [
                    {
                        name: 'x',
                        title: this.__('modAnnotateX'),
                        value: ff.x,
                        editable: true
                    },
                    {
                        name: 'y',
                        title: this.__('modAnnotateY'),
                        value: ff.y,
                        editable: true
                    },
                    {
                        name: 'w',
                        title: this.__('modAnnotateWidth'),
                        value: ff.w,
                        editable: true
                    },
                    {
                        name: 'h',
                        title: this.__('modAnnotateHeight'),
                        value: ff.h,
                        editable: true
                    }];

            case 'Circle':
                return [
                    {
                        name: 'x',
                        title: this.__('modAnnotateX'),
                        value: ff.x,
                        editable: true
                    },
                    {
                        name: 'y',
                        title: this.__('modAnnotateY'),
                        value: ff.y,
                        editable: true
                    },
                    {
                        name: 'radius',
                        title: this.__('modAnnotateRadius'),
                        value: ff.radius,
                        editable: true
                    }];
            default:
                return [];

        }

    }

    render() {
        let master = _master(this.props.controller);
        let f = this.props.annotateSelectedFeature;
        let ff = this.props.annotateFormData;

        let changed = (key, val) => this.props.controller.update({
            annotateFormData: {
                ...ff,
                [key]: val
            }
        });

        let submit = () => {
            f.updateFromForm(this.props.annotateFormData);
        };

        let data = this.formAttributes(f, ff);

        data.push({
            name: 'labelTemplate',
            title: this.__('modAnnotateLabelEdit'),
            value: ff['labelTemplate'],
            type: 'text',
            editable: true
        });

        return <div className="modAnnotateFeatureDetails">
            <Form>
                <Row>
                    <Cell flex>
                        <gws.components.sheet.Editor
                            data={data}
                            whenChanged={changed}
                            whenEntered={submit}
                        />
                    </Cell>
                </Row>
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            className="cmpButtonFormOk"
                            tooltip={this.props.controller.__('modAnnotateSaveButton')}
                            whenTouched={submit}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            className="cmpButtonFormCancel"
                            whenTouched={() => {
                                master.unselectFeature();
                                master.app.stopTool('Tool.Annotate.Modify');
                            }}
                        />
                    </Cell>
                </Row>
            </Form>
        </div>
    }
}

class AnnotateFeatureDetails extends gws.View<AnnotateViewProps> {
    render() {
        let master = _master(this.props.controller),
            layer = master.layer,
            selectedFeature = this.props.annotateSelectedFeature;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modAnnotateSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <AnnotateFeatureDetailsForm {...this.props} />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                    <Cell>
                        <gws.components.feature.TaskButton
                            controller={this.props.controller}
                            feature={selectedFeature}
                            source="annotate"
                        />
                    </Cell>

                    <sidebar.AuxButton
                        className="modAnnotateRemoveAuxButton"
                        tooltip={this.__('modAnnotateRemoveAuxButton')}
                        whenTouched={() => master.removeFeature(selectedFeature)}
                    />
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }

}

class AnnotateFeatureList extends gws.View<AnnotateViewProps> {
    render() {
        let master = _master(this.props.controller),
            layer = master.layer,
            selectedFeature = this.props.annotateSelectedFeature,
            features = layer ? layer.features : null;

        if (gws.tools.empty(features)) {
            return <sidebar.EmptyTab>
                {this.__('modAnnotateNotFound')}
            </sidebar.EmptyTab>;
        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modAnnotateSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.feature.List
                    controller={master}
                    features={features}
                    isSelected={f => f === selectedFeature}

                    content={(f: AnnotateFeature) => <gws.ui.Link
                        whenTouched={() => {
                            master.selectFeature(f, true);
                            master.app.startTool('Tool.Annotate.Modify')
                        }}
                        content={f.label}
                    />}

                    withZoom
                />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                    <sidebar.AuxButton
                        {...gws.tools.cls('modAnnotateEditAuxButton', this.props.appActiveTool === 'Tool.Annotate.Modify' && 'isActive')}
                        tooltip={this.__('modAnnotateEditAuxButton')}
                        whenTouched={() => master.app.startTool('Tool.Annotate.Modify')}
                    />
                    <sidebar.AuxButton
                        {...gws.tools.cls('modAnnotateDrawAuxButton', this.props.appActiveTool === 'Tool.Annotate.Draw' && 'isActive')}
                        tooltip={this.__('modAnnotateDrawAuxButton')}
                        whenTouched={() => master.app.startTool('Tool.Annotate.Draw')}
                    />
                    <sidebar.AuxButton
                        {...gws.tools.cls('modAnnotateClearAuxButton')}
                        tooltip={this.props.controller.__('modAnnotateClearAuxButton')}
                        whenTouched={() => master.clear()}
                    />
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }

}

class AnnotateSidebarView extends gws.View<AnnotateViewProps> {
    render() {
        if (this.props.annotateSelectedFeature) {
            return <AnnotateFeatureDetails {...this.props}/>;
        }
        return <AnnotateFeatureList {...this.props}/>;

    }
}

class AnnotateSidebar extends gws.Controller implements gws.types.ISidebarItem {

    iconClass = 'modAnnotateSidebarIcon';

    get tooltip() {
        return this.__('modAnnotateSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(AnnotateSidebarView, AnnotateStoreKeys)
        );
    }

}

class AnnotateDrawToolbarButton extends toolbar.Button {
    iconClass = 'modAnnotateDrawToolbarButton';
    tool = 'Tool.Annotate.Draw';

    get tooltip() {
        return this.__('modAnnotateDrawToolbarButton');
    }
}

class AnnotateController extends gws.Controller {
    uid = MASTER;
    layer: AnnotateLayer;
    modifyTool: AnnotateModifyTool;

    async init() {
        await super.init();

        await this.app.addTool('Tool.Annotate.Modify', this.modifyTool = this.app.createController(AnnotateModifyTool, this));
        await this.app.addTool('Tool.Annotate.Draw', this.app.createController(AnnotateDrawTool, this));
        this.layer = this.map.addServiceLayer(new AnnotateLayer(this.map, {
            uid: '_annotate',
        }));

        this.app.whenCalled('annotateFromFeature', args => {
            let f = this.newFromFeature(args.feature);

            if (f) {
                this.addAndFocus(f);
                this.update({sidebarActiveTab: 'Sidebar.Annotate'})
            }
        });

    }

    startLens() {
        let sel = this.getValue('annotateSelectedFeature') as gws.types.IMapFeature;
        if (sel) {
            this.update({
                lensGeometry: sel.geometry['clone']()
            })
            this.app.startTool('Tool.Lens');
        }

    }

    clear() {
        if (this.layer)
            this.map.removeLayer(this.layer);
        this.layer = null;
    }

    newFeature(shapeType, oFeature?: ol.Feature) {
        let
            sel = '.modAnnotate' + shapeType,
            style = this.map.getStyleFromSelector(sel),
            selectedStyle = this.map.getStyleFromSelector(sel + 'Selected');

        return new AnnotateFeature(_master(this), {
            shapeType,
            oFeature,
            style,
            selectedStyle,
            labelTemplate: defaultLabelTemplates[shapeType],
        });

    }

    addAndFocus(f: gws.types.IMapFeature) {
        this.layer.addFeature(f);
        this.selectFeature(f, false);
        this.app.startTool('Tool.Annotate.Modify')

    }

    newFromFeature(f: gws.types.IMapFeature) {
        let geometry = f.geometry;

        if (geometry) {
            let oFeature = new ol.Feature({geometry: geometry['clone']()});
            return this.newFeature(geometry.getType(), oFeature);
        }
    }

    selectFeature(f, highlight) {
        this.layer.features.forEach(f => (f as AnnotateFeature).setSelected(false));
        f.setSelected(true);

        this.update({
            annotateSelectedFeature: f,
            annotateFormData: f.formData,
        });

        if (highlight) {
            this.update({
                marker: {
                    features: [f],
                    mode: 'pan',
                }
            });
            f.oFeature.changed();
        } else {
            f.redraw();
        }

    }

    unselectFeature() {
        if (this.layer)
            this.layer.features.forEach(f => (f as AnnotateFeature).setSelected(false));
        this.update({
            annotateSelectedFeature: null,
            annotateFormData: {},
        });
    }

    featureUpdated(f) {
        let sel = this.getValue('annotateSelectedFeature');
        if (f === sel)
            this.update({
                annotateFormData: f.formData,
            });
    }

    removeFeature(f) {
        this.app.stopTool('Tool.Annotate.*');
        this.unselectFeature();
        this.layer.removeFeature(f);
        if (this.layer.features.length > 0)
            this.app.startTool('Tool.Annotate.Modify');

    }

}

export const tags = {
    [MASTER]: AnnotateController,
    'Sidebar.Annotate': AnnotateSidebar,
    'Toolbar.Annotate.Draw': AnnotateDrawToolbarButton,

};
