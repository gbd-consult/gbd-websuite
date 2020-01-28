import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as measure from 'gws/map/measure';
import * as style from 'gws/map/style';

import * as sidebar from './common/sidebar';
import * as toolbar from './common/toolbar';
import * as modify from './common/modify';
import * as draw from './common/draw';
import * as storage from './common/storage';

import {StyleController} from './style';

const MASTER = 'Shared.Annotate';
const STORAGE_CATEGORY = 'annotate';

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

interface ViewProps extends gws.types.ViewProps {
    controller: AnnotateController;
    annotateSelectedFeature: AnnotateFeature;
    annotateFormData: AnnotateFormData;
    annotateTab: string;
    annotateFeatureCount: number;
    appActiveTool: string;
};

const StoreKeys = [
    'annotateSelectedFeature',
    'annotateFormData',
    'annotateTab',
    'annotateUpdated',
    'appActiveTool',
];

function computeDimensionsWithMode(shapeType, geom, projection, mode) {
    let extent = geom.getExtent();

    if (shapeType === 'Point') {
        let c = (geom as ol.geom.Point).getCoordinates();
        return {
            x: c[0],
            y: c[1],
            extent,
        }
    }

    if (shapeType === 'Line') {
        return {
            len: measure.length(geom, projection, mode),
            extent,
        }
    }

    if (shapeType === 'Polygon') {
        return {
            len: measure.length(geom, projection, mode),
            area: measure.area(geom, projection, mode),
            extent,
        };
    }

    if (shapeType === 'Box') {

        let c = (geom as ol.geom.Polygon).getLinearRing(0).getCoordinates();

        // NB: x,y = top left
        return {
            len: measure.length(geom, projection, mode),
            area: measure.area(geom, projection, mode),
            x: c[0][0],
            y: c[3][1],
            w: measure.distance(c[0], c[1], projection, mode),
            h: measure.distance(c[1], c[2], projection, mode),
            extent,
        }
    }

    if (shapeType === 'Circle') {
        let g = geom as ol.geom.Circle;
        let c = g.getCenter();

        return {
            ...computeDimensions('Polygon', ol.geom.Polygon.fromCircle(g, 64, 0), projection),
            x: c[0],
            y: c[1],
            radius: g.getRadius(),
            extent,
        }
    }

    return {};
}

function computeDimensions(shapeType, geom, projection) {
    return computeDimensionsWithMode(shapeType, geom, projection, measure.ELLIPSOID)
}

function formatLengthForEdit(n) {
    return (Number(n) || 0).toFixed(2);
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
            case 'extent':
                return n.map(c => formatCoordinate(feature, c)).join(', ')

        }
    }

    function _length(n) {
        if (!n || n < 0.01)
            return '';
        if (n >= 1e3)
            return (n / 1e3).toFixed(2) + ' km';
        if (n > 1)
            return n.toFixed(2) + ' m';
        return n.toFixed(2) + ' m';
    }

    function _area(n) {
        let sq = '\u00b2';

        if (!n || n < 0.01)
            return '';
        if (n >= 1e5)
            return (n / 1e6).toFixed(2) + ' km' + sq;
        if (n > 1)
            return n.toFixed(2) + ' m' + sq;
        return n.toFixed(2) + ' m' + sq;
    }

    return (text || '').replace(/{(\w+)}/g, ($0, key) => _element(key));
}

function debugFeature(shapeType, geom, projection) {
    let fmt = new ol.format.WKT();

    let wkt = 'SRID=' + projection.getCode().split(':')[1] + ';' + fmt.writeGeometry(geom);
    let dims = [
        computeDimensionsWithMode(shapeType, geom, projection, measure.CARTESIAN),
        computeDimensionsWithMode(shapeType, geom, projection, measure.SPHERE),
        computeDimensionsWithMode(shapeType, geom, projection, measure.ELLIPSOID),
    ];

    return <pre>
        {wkt.replace(/,/g, ',\n')}
        <br/><br/>
        <h6>cartesian:</h6>{JSON.stringify(dims[0], null, 4)}<br/><br/>
        <h6>sphere   :</h6>{JSON.stringify(dims[1], null, 4)}<br/><br/>
        <h6>ellipsoid:</h6>{JSON.stringify(dims[2], null, 4)}
    </pre>;

}

class AnnotateFeature extends gws.map.Feature {
    labelTemplate: string;
    shapeType: string;

    get dimensions() {
        return computeDimensions(this.shapeType, this.geometry, this.map.projection);
    }

    getProps() {
        let p = super.getProps();
        p.elements = {
            shapeType: this.shapeType,
            labelTemplate: this.labelTemplate,
        }
        return p;

    }

    get formData(): AnnotateFormData {
        let dims = this.dimensions;
        return {
            x: formatCoordinate(this, dims.x),
            y: formatCoordinate(this, dims.y),
            radius: formatLengthForEdit(dims.radius),
            w: formatLengthForEdit(dims.w),
            h: formatLengthForEdit(dims.h),
            labelTemplate: this.labelTemplate,
        }
    }

    get master(): AnnotateController {
        return this.map.app.controller(MASTER) as AnnotateController;
    }

    constructor(map, args: gws.types.IMapFeatureArgs) {
        super(map, args);

        this.labelTemplate = args.props.elements['labelTemplate'];
        this.shapeType = args.props.elements['shapeType'];

        //this.oFeature.on('change', e => this.onChange(e));
        this.geometry.on('change', e => this.redraw());
    }

    setStyles(src) {
        let styleNames = this.master.app.style.getMap(src);
        let normal = this.master.app.style.at(styleNames.normal);
        let selName = '_selected_' + normal.name;
        let selected = this.master.app.style.at(selName) || this.master.app.style.add(
            new style.CascadedStyle(selName, [normal, this.master.selectedStyle]));
        super.setStyles({normal, selected});
    }

    setChanged() {
        super.setChanged();
        this.geometry.changed();
    }

    setSelected(sel) {
        this.setMode(sel ? 'selected' : 'normal');
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

        this.setChanged();

    }

}

class AnnotateLayer extends gws.map.layer.FeatureLayer {
}

class AnnotateDrawTool extends draw.Tool {
    drawFeature: AnnotateFeature;
    styleName = '.modAnnotateDraw';

    get title() {
        return this.__('modAnnotateDrawToolbarButton')
    }

    start() {
        super.start();
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Annotate'});
    }

    whenStarted(shapeType, oFeature) {
        this.drawFeature = _master(this).newFeature(shapeType, oFeature);
    }

    whenEnded() {
        _master(this).addAndSelectFeature(this.drawFeature);
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
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Annotate'});

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

class AnnotateFeatureForm extends gws.View<ViewProps> {

    render() {
        let cc = _master(this.props.controller);

        let selectedFeature = this.props.annotateSelectedFeature,
            st = selectedFeature.shapeType;

        let submit = () => {
            selectedFeature.updateFromForm(this.props.annotateFormData);
            cc.unselectFeature();
            cc.app.stopTool('Tool.Annotate.Modify');
        };


        let form = [];

        if (['Point', 'Box', 'Circle'].includes(st)) {
            form.push(<gws.ui.NumberInput label={this.__('modAnnotateX')} {...cc.bind('annotateFormData.x')}/>)
            form.push(<gws.ui.NumberInput label={this.__('modAnnotateY')} {...cc.bind('annotateFormData.y')}/>)
        }

        if (st === 'Box') {
            form.push(<gws.ui.NumberInput label={this.__('modAnnotateWidth')} {...cc.bind('annotateFormData.w')}/>)
            form.push(<gws.ui.NumberInput label={this.__('modAnnotateHeight')} {...cc.bind('annotateFormData.h')}/>)
        }

        if (st === 'Circle') {
            form.push(<gws.ui.NumberInput
                label={this.__('modAnnotateRadius')} {...cc.bind('annotateFormData.radius')}/>)
        }

        form.push(<gws.ui.TextArea
            label={this.__('modAnnotateLabelEdit')}
            {...cc.bind('annotateFormData.labelTemplate')}
        />);

        return <div className="modAnnotateFeatureDetails">
            <Form tabular children={form}/>
            <Form>
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
                            className="modAnnotateStyleButton"
                            tooltip={this.props.controller.__('modAnnotateSaveButton')}
                            whenTouched={() => cc.update({annotateTab: 'style'})}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.Button
                            className="modAnnotateRemoveButton"
                            tooltip={this.__('modAnnotateRemoveButton')}
                            whenTouched={() => cc.removeFeature(selectedFeature)}
                        />
                    </Cell>
                </Row>
            </Form>
        </div>
    }
}

class AnnotateFeatureTabFooter extends gws.View<ViewProps> {
    render() {
        let cc = _master(this.props.controller),
            selectedFeature = this.props.annotateSelectedFeature,
            tab = this.props.annotateTab === 'style' ? 'style' : 'form';

        let close = () => {
            cc.unselectFeature();
            cc.app.stopTool('Tool.Annotate.Modify');
        };

        return <sidebar.TabFooter>
            <sidebar.AuxToolbar>
                <Cell>
                    <sidebar.AuxButton
                        {...gws.tools.cls('modAnnotateFormAuxButton', tab === 'form' && 'isActive')}
                        whenTouched={() => cc.update({annotateTab: 'form'})}
                        tooltip={cc.__('modAnnotateFormAuxButton')}
                    />
                </Cell>
                <Cell>
                    <sidebar.AuxButton
                        {...gws.tools.cls('modAnnotateStyleAuxButton', tab === 'style' && 'isActive')}
                        whenTouched={() => cc.update({annotateTab: 'style'})}
                        tooltip={cc.__('modAnnotateStyleAuxButton')}
                    />
                </Cell>
                <Cell flex/>
                <Cell>
                    <gws.components.feature.TaskButton
                        controller={this.props.controller}
                        feature={selectedFeature}
                        source="annotate"
                    />
                </Cell>
                <Cell>
                    <sidebar.AuxCloseButton
                        whenTouched={close}
                    />
                </Cell>
            </sidebar.AuxToolbar>
        </sidebar.TabFooter>
    }
}

class AnnotateFeatureFormTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this.props.controller),
            selectedFeature = this.props.annotateSelectedFeature;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modAnnotateSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <AnnotateFeatureForm {...this.props} />
                {selectedFeature.labelTemplate.indexOf('debug') > 0 && debugFeature(selectedFeature.shapeType, selectedFeature.geometry, selectedFeature.map.projection)}
            </sidebar.TabBody>

            <AnnotateFeatureTabFooter {...this.props}/>
        </sidebar.Tab>
    }
}

class AnnotateFeatureStyleTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this.props.controller),
            sc = cc.app.controller('Shared.Style') as StyleController;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modAnnotateSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {sc.styleForm()}
            </sidebar.TabBody>

            <AnnotateFeatureTabFooter {...this.props}/>
        </sidebar.Tab>
    }

}

class AnnotateListTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this.props.controller),
            selectedFeature = this.props.annotateSelectedFeature;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modAnnotateSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {cc.hasFeatures
                    ? <gws.components.feature.List
                        controller={cc}
                        features={cc.features}
                        isSelected={f => f === selectedFeature}

                        content={(f: AnnotateFeature) => <gws.ui.Link
                            whenTouched={() => {
                                cc.app.stopTool('Tool.Annotate.Modify');
                                cc.selectFeature(f, true);
                                cc.app.startTool('Tool.Annotate.Modify');
                            }}
                            content={f.label}
                        />}

                        rightButton={f => <gws.components.list.Button
                            className="modAnnotateDeleteListButton"
                            whenTouched={() => cc.removeFeature(f)}
                        />}

                        withZoom
                    />
                    : <sidebar.EmptyTabBody>
                        {this.__('modAnnotateNotFound')}
                    </sidebar.EmptyTabBody>
                }
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        {...gws.tools.cls('modAnnotateEditAuxButton', this.props.appActiveTool === 'Tool.Annotate.Modify' && 'isActive')}
                        tooltip={this.__('modAnnotateEditAuxButton')}
                        disabled={!cc.hasFeatures}
                        whenTouched={() => cc.app.startTool('Tool.Annotate.Modify')}
                    />
                    <sidebar.AuxButton
                        {...gws.tools.cls('modAnnotateAddAuxButton')}
                        tooltip={this.props.controller.__('modAnnotateAddAuxButton')}
                        whenTouched={() => cc.app.toggleTool('Tool.Annotate.Draw')}
                    />
                    <Cell flex/>
                    <storage.ReadAuxButton
                        controller={cc}
                        category={STORAGE_CATEGORY}
                        whenDone={data => cc.storageSetter(data)}
                    />
                    {<storage.WriteAuxButton
                        controller={this.props.controller}
                        category={STORAGE_CATEGORY}
                        disabled={!cc.hasFeatures}
                        data={cc.storageGetter()}
                    />}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }

}

class AnnotateSidebarView extends gws.View<ViewProps> {
    render() {
        if (!this.props.annotateSelectedFeature) {
            return <AnnotateListTab {...this.props}/>;
        }
        if (this.props.annotateTab === 'style') {
            return <AnnotateFeatureStyleTab {...this.props}/>;
        }
        return <AnnotateFeatureFormTab {...this.props}/>;
    }
}

class AnnotateSidebar extends gws.Controller implements gws.types.ISidebarItem {

    iconClass = 'modAnnotateSidebarIcon';

    get tooltip() {
        return this.__('modAnnotateSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(AnnotateSidebarView, StoreKeys)
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
    selectedStyle: gws.types.IStyle;

    async init() {
        await super.init();

        let selectedSelector = '.modAnnotateSelected';

        this.selectedStyle = this.app.style.add(new style.Style(
            selectedSelector,
            style.valuesFromCssSelector(selectedSelector)));

        this.layer = this.map.addServiceLayer(new AnnotateLayer(this.map, {
            uid: '_annotate',
        }));

        this.update({
            annotateLastStyleName: '.modAnnotateFeature',
        });

        // this.app.whenChanged('annotateFormData', () => {
        //     let f = this.getValue('annotateSelectedFeature');
        //     if (f)
        //         f.updateFromForm(this.getValue('annotateFormData'));
        // })

        this.app.whenCalled('annotateFromFeature', args => {
            let f = this.newFromFeature(args.feature);
            if (f) {
                this.addAndSelectFeature(f);
                this.update({sidebarActiveTab: 'Sidebar.Annotate'})
            }
        });
    }

    get features(): Array<AnnotateFeature> {
        return this.layer ? this.layer.features : [];
    }

    get hasFeatures() {
        return this.features.length > 0;
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

    newFromFeature(f: gws.types.IMapFeature) {

        function singleGeometry(geom) {
            if (!geom)
                return null;

            let gt = geom.getType();

            if (gt === 'MultiPolygon') {
                return (geom as ol.geom.MultiPolygon).getPolygon(0);
            }
            if (gt === 'MultiLineString') {
                return (geom as ol.geom.MultiLineString).getLineString(0);
            }

            if (gt === 'MultiPoint') {
                return (geom as ol.geom.MultiPoint).getPoint(0);
            }

            return geom;
        }


        let geometry = singleGeometry(f.geometry);

        if (geometry) {
            let oFeature = new ol.Feature({geometry: geometry['clone']()});
            let shapeType = geometry.getType();
            if (shapeType === 'LineString')
                shapeType = 'Line';
            return this.newFeature(shapeType, oFeature);
        }
    }

    newFeature(shapeType, oFeature?: ol.Feature) {
        let sty = this.app.style.at(this.getValue('annotateLastStyleName'));

        let newStyle = this.app.style.add(new style.Style(
            gws.tools.uniqId('AnnotateStyle'),
            sty.values));

        let feat = new AnnotateFeature(this.map, {
            oFeature,
            props: {
                uid: gws.tools.uniqId('annotate'),
                elements: {
                    master: this,
                    shapeType,
                    labelTemplate: defaultLabelTemplates[shapeType],
                }
            },
            style: newStyle
        });

        this.update({
            annotateLastStyleName: newStyle.name,
        });

        return feat;
    }

    addAndSelectFeature(f: gws.types.IMapFeature) {
        this.addFeature(f);
        this.selectFeature(f, false);
        this.app.startTool('Tool.Annotate.Modify')

    }

    selectFeature(f, panTo) {
        if (panTo) {
            this.update({
                marker: {
                    features: [f],
                    mode: 'pan',
                }
            });
        }

        this.features.forEach(f => f.setSelected(false));
        f.setSelected(true);

        this.update({
            annotateSelectedFeature: f,
            annotateFormData: f.formData,
            styleEditorCurrentName: f.styleNames.normal,
        });

    }

    unselectFeature() {
        this.features.forEach(f => f.setSelected(false));
        this.update({
            annotateSelectedFeature: null,
            annotateFormData: {},
            styleEditorCurrentName: '',
        });
    }

    featureUpdated(f) {
        let sel = this.getValue('annotateSelectedFeature');
        if (f === sel)
            this.update({
                annotateFormData: f.formData,
            });
    }


    addFeature(f) {
        this.layer.addFeature(f);
        this.setUpdated();
    }

    removeFeature(f) {
        this.app.stopTool('Tool.Annotate.*');
        this.unselectFeature();
        this.layer.removeFeature(f);
        if (this.hasFeatures)
            this.app.startTool('Tool.Annotate.Modify');
        this.setUpdated();
    }

    storageSetter(data) {
        let lastFeature = null;

        this.clear();

        this.layer.addFeatures(data.features.map(props => {
            let f = new AnnotateFeature(this.map, {props})
            f.setChanged();
            lastFeature = f;
            return f;
        }));

        if (this.hasFeatures)
            this.app.startTool('Tool.Annotate.Modify');

        if (lastFeature) {
            this.update({
                annotateLastStyleName: lastFeature.styleNames.normal,
            });
        }

        this.setUpdated();
    }

    storageGetter() {
        return {
            'features': this.features.map(f => f.getProps())
        }
    }

    clear() {
        this.app.stopTool('Tool.Annotate.*');
        this.layer.clear();
        this.setUpdated();
    }

    setUpdated() {
        this.update({
            annotateUpdated: (this.getValue('annotateUpdated') || 0) + 1,
        });
    }


}

export const tags = {
    [MASTER]: AnnotateController,
    'Sidebar.Annotate': AnnotateSidebar,
    'Toolbar.Annotate.Draw': AnnotateDrawToolbarButton,
    'Tool.Annotate.Modify': AnnotateModifyTool,
    'Tool.Annotate.Draw': AnnotateDrawTool,

};
