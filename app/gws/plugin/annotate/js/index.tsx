import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as measure from 'gws/map/measure';
import * as template from 'gws/map/template';
import * as style from 'gws/map/style';
import * as styler from 'gws/elements/styler';
import * as draw from 'gws/elements/draw';
import * as modify from 'gws/elements/modify';
import * as sidebar from 'gws/elements/sidebar';
import * as toolbar from 'gws/elements/toolbar';
import * as components from 'gws/components';


const MASTER = 'Shared.Annotate';
const STORAGE_CATEGORY = 'Annotate';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as AnnotateController;

let {Form, Row, Cell} = gws.ui.Layout;

const defaultLabelTemplates = {
    Point: '{xy}',
    Line: '{len}',
    Polygon: '{area}',
    Circle: '{radius}',
    Box: '{width} x {height}',
};

interface AnnotateFormData {
    x?: string
    y?: string
    width?: string
    height?: string
    radius?: string
    labelTemplate: string
    shapeType: string
}

interface ViewProps extends gws.types.ViewProps {
    controller: AnnotateController;
    annotateFeatures: Array<AnnotateFeature>;
    annotateSelectedFeature: AnnotateFeature;
    annotateFormData: AnnotateFormData;
    annotateTab: string;
    annotateFeatureCount: number;
    annotateLabelTemplates: gws.types.Dict;
    appActiveTool: string;
};

const StoreKeys = [
    'annotateFeatures',
    'annotateSelectedFeature',
    'annotateFormData',
    'annotateTab',
    'annotateUpdated',
    'annotateLabelTemplates',
    'appActiveTool',
];


// function debugFeature(shapeType, geom, projection) {
//     let fmt = new ol.format.WKT();
//
//     let wkt = 'SRID=' + projection.getCode().split(':')[1] + ';' + fmt.writeGeometry(geom);
//     let dims = [
//         computeDimensionsWithMode(shapeType, geom, projection, measure.Mode.CARTESIAN),
//         computeDimensionsWithMode(shapeType, geom, projection, measure.Mode.SPHERE),
//         computeDimensionsWithMode(shapeType, geom, projection, measure.Mode.ELLIPSOID),
//     ];
//
//     return <pre>
//         {wkt.replace(/,/g, ',\n')}
//         <br/><br/>
//         <h6>cartesian:</h6>{JSON.stringify(dims[0], null, 4)}<br/><br/>
//         <h6>sphere   :</h6>{JSON.stringify(dims[1], null, 4)}<br/><br/>
//         <h6>ellipsoid:</h6>{JSON.stringify(dims[2], null, 4)}
//     </pre>;
//
// }

class AnnotateFeature extends gws.map.Feature {
    labelTemplate: string;
    shapeType: string;

    getProps() {
        let p = super.getProps();
        p.elements = {
            shapeType: this.shapeType,
            labelTemplate: this.labelTemplate,
            label: this.format(this.labelTemplate),
        };
        return p;
    }

    get formData(): AnnotateFormData {
        return {
            shapeType: this.shapeType,
            labelTemplate: this.labelTemplate,
            x: this.format('{x|M|2}'),
            y: this.format('{y|M|2}'),
            radius: this.format('{radius|M|2}'),
            width: this.format('{width|M|2}'),
            height: this.format('{height|M|2}'),
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
        this.setLabel(this.format(this.labelTemplate));
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
                width = Number(ff.width) || 100,
                height = Number(ff.height) || 100;

            let
                a = [x, y] as ol.Coordinate,
                b = measure.direct(a, 180, height, this.map.projection, measure.Mode.ELLIPSOID),
                c = measure.direct(b, 90, width, this.map.projection, measure.Mode.ELLIPSOID),
                d = measure.direct(c, 0, height, this.map.projection, measure.Mode.ELLIPSOID);

            g.setCoordinates([[a, b, c, d, a]]);
        }

        this.setChanged();
    }

    protected format(text) {
        return template.formatGeometry(text, this.geometry, this.map.projection);
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

    get defaultPlaceholders() {
        let pcb = ['Polygon', 'Circle', 'Box'];

        return [
            {shapeTypes: pcb, text: this.__('mapPlaceholderArea') + ' (10 m\u00b2)', value: '{area | m}'},
            {shapeTypes: pcb, text: this.__('mapPlaceholderArea') + ' (10.99 m\u00b2)', value: '{area | m | 2}'},
            {shapeTypes: pcb, text: this.__('mapPlaceholderArea') + ' (10 km\u00b2)', value: '{area | km}'},
            {shapeTypes: pcb, text: this.__('mapPlaceholderArea') + ' (10.99 km\u00b2)', value: '{area | km | 2}'},
            {shapeTypes: pcb, text: this.__('mapPlaceholderArea') + ' (10 ha)', value: '{area | ha}'},
            {shapeTypes: pcb, text: this.__('mapPlaceholderArea') + ' (10.99 ha)', value: '{area | ha | 2}'},

            {shapeTypes: ['Line'], text: this.__('mapPlaceholderLength') + ' (10 m)', value: '{len | m}'},
            {shapeTypes: ['Line'], text: this.__('mapPlaceholderLength') + ' (10.99 m)', value: '{len | m | 2}'},
            {shapeTypes: ['Line'], text: this.__('mapPlaceholderLength') + ' (10 km)', value: '{len | km}'},
            {shapeTypes: ['Line'], text: this.__('mapPlaceholderLength') + ' (10.99 km)', value: '{len | km | 2}'},

            {shapeTypes: pcb, text: this.__('mapPlaceholderPerimeter') + ' (10 m)', value: '{len | m}'},
            {shapeTypes: pcb, text: this.__('mapPlaceholderPerimeter') + ' (10.99 m)', value: '{len | m | 2}'},
            {shapeTypes: pcb, text: this.__('mapPlaceholderPerimeter') + ' (10 km)', value: '{len | km}'},
            {shapeTypes: pcb, text: this.__('mapPlaceholderPerimeter') + ' (10.99 km)', value: '{len | km | 2}'},

            {shapeTypes: ['Circle'], text: this.__('mapPlaceholderRadius') + ' (10 m)', value: '{radius | m}'},
            {shapeTypes: ['Circle'], text: this.__('mapPlaceholderRadius') + ' (10.99 m)', value: '{radius | m | 2}'},
            {shapeTypes: ['Circle'], text: this.__('mapPlaceholderRadius') + ' (10 km)', value: '{radius | km}'},
            {shapeTypes: ['Circle'], text: this.__('mapPlaceholderRadius') + ' (10.99 km)', value: '{radius | km | 2}'},

            {shapeTypes: ['Box'], text: this.__('mapPlaceholderWidth') + ' (10 m)', value: '{width | m}'},
            {shapeTypes: ['Box'], text: this.__('mapPlaceholderWidth') + ' (10.99 m)', value: '{width | m | 2}'},
            {shapeTypes: ['Box'], text: this.__('mapPlaceholderWidth') + ' (10 km)', value: '{width | km}'},
            {shapeTypes: ['Box'], text: this.__('mapPlaceholderWidth') + ' (10.99 km)', value: '{width | km | 2}'},

            {shapeTypes: ['Box'], text: this.__('mapPlaceholderHeight') + ' (10 m)', value: '{height | m}'},
            {shapeTypes: ['Box'], text: this.__('mapPlaceholderHeight') + ' (10.99 m)', value: '{height | m | 2}'},
            {shapeTypes: ['Box'], text: this.__('mapPlaceholderHeight') + ' (10 km)', value: '{height | km}'},
            {shapeTypes: ['Box'], text: this.__('mapPlaceholderHeight') + ' (10.99 km)', value: '{height | km | 2}'},

            {shapeTypes: ['Point'], text: this.__('mapPlaceholderXY') + ' (m)', value: '{xy | m | 2}'},
            {shapeTypes: ['Point'], text: this.__('mapPlaceholderXY') + ' (deg)', value: '{xy | deg | 5}'},
            {shapeTypes: ['Point'], text: this.__('mapPlaceholderXY') + ' (dms)', value: '{xy | dms}'},
        ];
    }

    render() {
        let cc = _master(this.props.controller);

        let selectedFeature = this.props.annotateSelectedFeature,
            st = selectedFeature.shapeType;

        let submit = () => {
            selectedFeature.updateFromForm(this.props.annotateFormData);
            cc.unselectFeature();
            cc.app.stopTool('Tool.Annotate.Modify');
        };

        let placeholders = this.defaultPlaceholders
            .filter(p => p.shapeTypes.includes(st))
            .map(p => ({text: p.text, value: p.value}));

        let labelEditorRef: React.RefObject<HTMLTextAreaElement> = React.createRef();

        let insertPlacehodler = p => {
            let ta = labelEditorRef.current;
            if (!ta)
                return;
            let
                val = ta.value || '',
                s = ta.selectionStart || 0,
                e = ta.selectionEnd || 0;

            val = val.slice(0, s) + p + val.slice(e);
            ta.focus();
            cc.updateObject('annotateFormData', {labelTemplate: val});
        };

        let form = [];

        let decimalFmt = {
            decimal: ","
        };

        if (['Point', 'Box', 'Circle'].includes(st)) {
            form.push(<gws.ui.NumberInput locale={this.props.controller.app.locale} label={this.__('modAnnotateX')} {...cc.bind('annotateFormData.x')}/>)
            form.push(<gws.ui.NumberInput locale={this.props.controller.app.locale} label={this.__('modAnnotateY')} {...cc.bind('annotateFormData.y')}/>)
        }

        if (st === 'Box') {
            form.push(<gws.ui.NumberInput locale={this.props.controller.app.locale} label={this.__('modAnnotateWidth')} {...cc.bind('annotateFormData.width')}/>)
            form.push(<gws.ui.NumberInput locale={this.props.controller.app.locale} label={this.__('modAnnotateHeight')} {...cc.bind('annotateFormData.height')}/>)
        }

        if (st === 'Circle') {
            form.push(<gws.ui.NumberInput locale={this.props.controller.app.locale} label={this.__('modAnnotateRadius')} {...cc.bind('annotateFormData.radius')}/>)
        }

        form.push(<gws.ui.TextArea
            focusRef={labelEditorRef}
            label={this.__('modAnnotateLabelEdit')}
            {...cc.bind('annotateFormData.labelTemplate')}
        />);

        form.push(<gws.ui.Select
            label={this.__('modAnnotatePlaceholder')}
            value=''
            items={placeholders}
            whenChanged={insertPlacehodler}
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
                            tooltip={this.props.controller.__('modAnnotateStyleButton')}
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
                        {...gws.lib.cls('modAnnotateFormAuxButton', tab === 'form' && 'isActive')}
                        whenTouched={() => cc.update({annotateTab: 'form'})}
                        tooltip={cc.__('modAnnotateFormAuxButton')}
                    />
                </Cell>
                <Cell>
                    <sidebar.AuxButton
                        {...gws.lib.cls('modAnnotateStyleAuxButton', tab === 'style' && 'isActive')}
                        whenTouched={() => cc.update({annotateTab: 'style'})}
                        tooltip={cc.__('modAnnotateStyleAuxButton')}
                    />
                </Cell>
                <Cell flex/>
                <Cell>
                    <components.feature.TaskButton
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
                <gws.ui.Title content={this.__('modAnnotateSidebarDetailsTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <AnnotateFeatureForm {...this.props} />
            </sidebar.TabBody>

            <AnnotateFeatureTabFooter {...this.props}/>
        </sidebar.Tab>
    }
}

class AnnotateFeatureStyleTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this.props.controller),
            sc = cc.app.controller('Shared.Style') as styler.Controller;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modAnnotateSidebarDetailsTitle')}/>
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
            selectedFeature = this.props.annotateSelectedFeature,
            hasFeatures = this.props.annotateFeatures && this.props.annotateFeatures.length > 0;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modAnnotateSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {cc.hasFeatures
                    ? <components.feature.List
                        controller={cc}
                        features={this.props.annotateFeatures || []}
                        isSelected={f => f === selectedFeature}

                        content={(f: AnnotateFeature) => <gws.ui.Link
                            whenTouched={() => {
                                cc.app.stopTool('Tool.Annotate.Modify');
                                cc.selectFeature(f, true);
                                cc.app.startTool('Tool.Annotate.Modify');
                            }}
                            content={f.label || '...'}
                        />}

                        rightButton={f => <components.list.Button
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
                        {...gws.lib.cls('modAnnotateEditAuxButton', this.props.appActiveTool === 'Tool.Annotate.Modify' && 'isActive')}
                        tooltip={this.__('modAnnotateEditAuxButton')}
                        disabled={!hasFeatures}
                        whenTouched={() => cc.app.startTool('Tool.Annotate.Modify')}
                    />
                    <sidebar.AuxButton
                        {...gws.lib.cls('modAnnotateAddAuxButton')}
                        tooltip={this.props.controller.__('modAnnotateAddAuxButton')}
                        whenTouched={() => cc.app.toggleTool('Tool.Annotate.Draw')}
                    />
                    <Cell flex/>
                    {/*{storage.auxButtons(cc, {*/}
                    {/*    category: STORAGE_CATEGORY,*/}
                    {/*    hasData: hasFeatures,*/}
                    {/*    getData: name => cc.storageGetData(name),*/}
                    {/*    dataReader: (name, data) => cc.storageReader(name, data)*/}
                    {/*})}*/}
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
        this.selectedStyle = this.app.style.get(selectedSelector);

        this.layer = this.map.addServiceLayer(new AnnotateLayer(this.map, {
            uid: '_annotate',
        }));

        this.update({
            annotateLastStyleName: '.modAnnotateFeature',
            annotateLabelTemplates: defaultLabelTemplates,
        });

        this.app.whenChanged('annotateFormData', data => {
            this.updateObject('annotateLabelTemplates', {
                [data.shapeType]: data.labelTemplate,
            });
            // let f = this.getValue('annotateSelectedFeature');
            // if (f)
            //     f.updateFromForm(this.getValue('annotateFormData'));
        })

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
            gws.lib.uniqId('AnnotateStyle'),
            sty.values));

        let templates = this.getValue('annotateLabelTemplates'),
            labelTemplate = (templates && templates[shapeType]) || defaultLabelTemplates[shapeType];

        let feat = new AnnotateFeature(this.map, {
            oFeature,
            props: {
                uid: gws.lib.uniqId('annotate'),
                attributes: {},
                modelUid: '',
                elements: {
                    shapeType,
                    labelTemplate,
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
        f.setChanged();

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

    storageReader(name, data) {
        let lastFeature = null;

        this.app.stopTool('Tool.Annotate.*');
        this.layer.clear();

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

    storageGetData(name) {
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
            annotateFeatures: this.features,
        });
    }


}

gws.registerTags({
    [MASTER]: AnnotateController,
    'Sidebar.Annotate': AnnotateSidebar,
    'Toolbar.Annotate.Draw': AnnotateDrawToolbarButton,
    'Tool.Annotate.Modify': AnnotateModifyTool,
    'Tool.Annotate.Draw': AnnotateDrawTool,

});
