import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as template from 'gws/map/template';
import * as measure from 'gws/map/measure';
import * as style from 'gws/map/style';

import * as sidebar from './sidebar';
import * as toolbar from './toolbar';
import * as draw from './draw';
import * as storage from './storage';

import {StyleController} from './style';

const MASTER = 'Shared.Annotate';
const STORAGE_CATEGORY = 'Annotate';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

let {Form, Row, Cell} = gws.ui.Layout;

const defaultLabelTemplates = {
    Point: '{xy}',
    Line: '{len | m | 2}',
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
    controller: Controller;
    annotateSelectedFeature: AnnotateFeature;
    annotateFormData: AnnotateFormData;
    annotateTab: string;
    annotateFeatureCount: number;
    annotateLabelTemplates: gws.types.Dict;
    appActiveTool: string;
    mapFocusedFeature: gws.types.IFeature;
    mapUpdateCount: number;
}

const StoreKeys = [
    'annotateSelectedFeature',
    'annotateFormData',
    'annotateTab',
    'annotateUpdated',
    'annotateLabelTemplates',
    'appActiveTool',
    'mapFocusedFeature',
    // 'mapUpdateCount',
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

    getProps(depth?) {
        let p = super.getProps(depth);
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


    // setProps(props) {
    //     super.setProps(props);
    //
    //     this.labelTemplate = props.elements['labelTemplate'];
    //     this.shapeType = props.elements['shapeType'];
    //
    //     //this.oFeature.on('change', e => this.onChange(e));
    //     // this.geometry.on('change', e => this.redraw());
    //
    //     return this;
    // }

    // setStyles(src) {
    //     let styleNames = this.master.app.style.getMap(src);
    //     let normal = this.master.app.style.at(styleNames.normal);
    //     let selName = '_selected_' + normal.name;
    //     let selected = this.master.app.style.at(selName) || this.master.app.style.add(
    //         new style.CascadedStyle(selName, [normal, this.master.selectedStyle]));
    //     super.setStyles({normal, selected});
    // }

    // setChanged() {
    //     super.setChanged();
    //     this.geometry.changed();
    // }

    whenGeometryChanged() {
        this.update();
    }


    update() {
        this.elements.label = this.format(this.labelTemplate);

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

        this.update();
    }

    protected format(text) {
        return template.formatGeometry(text, this.geometry, this.map.projection);
    }

}


class AnnotateDrawTool extends draw.Tool {
    drawFeature: AnnotateFeature;
    styleName = '.modAnnotateDraw';

    get title() {
        return this.__('modAnnotateDrawToolbarButton')
    }

    enabledShapes() {
        return [
            'Point',
            'Line',
            'Polygon',
            'Circle',
        ]
    }

    whenStarted(shapeType, oFeature) {
        let cc = _master(this);
        this.drawFeature = cc.newFeatureFromDraw(shapeType, oFeature);
        cc.map.focusFeature(null);
    }

    whenEnded(shapeType, oFeature) {
        this.map.focusFeature(this.drawFeature);

    }

    whenCancelled() {
        _master(this).removeFeature(this.drawFeature);
    }
}

export class AnnotateModifyTool extends gws.Tool {
    oFeatureCollection: ol.Collection<ol.Feature>;
    snap: boolean = true;

    setEditable(feature?: gws.types.IFeature) {
        if (this.oFeatureCollection) {
            this.oFeatureCollection.clear();
            if (feature)
                this.oFeatureCollection.push(feature.oFeature);
        }
    }

    whenTouched(evt: ol.MapBrowserEvent) {
        let found = null;
        let cc = _master(this);

        cc.map.oMap.forEachFeatureAtPixel(evt.pixel, oFeature => {
            if (found)
                return;

            let feature = oFeature['_gwsFeature'];
            if (feature && feature.layer === cc.layer) {
                found = feature;
            }
        });

        if (found) {
            cc.map.focusFeature(found);
            this.setEditable(found);
            return true;
        }

        cc.map.focusFeature(null);
        this.setEditable(null);

    }


    start() {
        let cc = _master(this);

        this.oFeatureCollection = new ol.Collection<ol.Feature>();

        let opts = {
            handleEvent: (evt: ol.MapBrowserEvent) => {
                if (evt.type === 'singleclick') {
                    this.whenTouched(evt);
                    return false;
                }
                return true
            },
        };

        let ixPointer = new ol.interaction.Pointer(opts);

        let ixModify = this.map.modifyInteraction({
            features: this.oFeatureCollection,
            whenEnded: oFeatures => {
                if (oFeatures[0]) {
                    let feature = oFeatures[0]['_gwsFeature'];
                    if (feature) {
                        feature.update();
                    }
                }
            }
        });

        let ixs: Array<ol.interaction.Interaction> = [ixPointer, ixModify];
        this.map.appendInteractions(ixs);
    }

    stop() {
        this.oFeatureCollection = null;
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
        let feature = cc.selectedFeature;
        let shapeType = feature.shapeType;

        let submit = () => {
            feature.updateFromForm(this.props.annotateFormData);
            cc.map.focusFeature(null);
            cc.app.stopTool('Tool.Annotate.Modify');
        };

        let placeholders = this.defaultPlaceholders
            .filter(p => p.shapeTypes.includes(shapeType))
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

        if (['Point', 'Box', 'Circle'].includes(shapeType)) {
            form.push(<gws.ui.NumberInput locale={cc.app.locale}
                                          label={this.__('modAnnotateX')} {...cc.bind('annotateFormData.x')}/>)
            form.push(<gws.ui.NumberInput locale={cc.app.locale}
                                          label={this.__('modAnnotateY')} {...cc.bind('annotateFormData.y')}/>)
        }

        if (shapeType === 'Box') {
            form.push(<gws.ui.NumberInput locale={cc.app.locale}
                                          label={this.__('modAnnotateWidth')} {...cc.bind('annotateFormData.width')}/>)
            form.push(<gws.ui.NumberInput locale={cc.app.locale}
                                          label={this.__('modAnnotateHeight')} {...cc.bind('annotateFormData.height')}/>)
        }

        if (shapeType === 'Circle') {
            form.push(<gws.ui.NumberInput locale={cc.app.locale}
                                          label={this.__('modAnnotateRadius')} {...cc.bind('annotateFormData.radius')}/>)
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
                            tooltip={cc.__('modAnnotateSaveButton')}
                            whenTouched={submit}
                        />
                    </Cell>
                    {<Cell>
                        <gws.ui.Button
                            className="modAnnotateStyleButton"
                            tooltip={cc.__('modAnnotateStyleButton')}
                            whenTouched={() => cc.update({annotateTab: 'style'})}
                        />
                    </Cell>}
                    <Cell>
                        <gws.ui.Button
                            className="modAnnotateRemoveButton"
                            tooltip={this.__('modAnnotateRemoveButton')}
                            whenTouched={() => cc.removeFeature(feature)}
                        />
                    </Cell>
                </Row>
            </Form>
        </div>
    }
}

class AnnotateFeatureTabFooter extends gws.View<ViewProps> {
    render() {
        let cc = _master(this),
            selectedFeature = this.props.annotateSelectedFeature,
            tab = this.props.annotateTab === 'style' ? 'style' : 'form';

        let close = () => {
            cc.map.focusFeature(null);
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
                {false && <Cell>
                    <gws.components.feature.TaskButton
                        controller={this.props.controller}
                        feature={selectedFeature}
                        source="annotate"
                    />
                </Cell>}
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
            sc = cc.app.controller('Shared.Style') as StyleController;

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
        let cc = _master(this);
        let features = cc.features;
        let selectedFeature = cc.selectedFeature;
        let tabBody;

        if (features.length > 0) {
            tabBody = <gws.components.feature.List
                controller={cc}
                features={features || []}
                isSelected={f => f === selectedFeature}

                content={(f: AnnotateFeature) => <gws.ui.Link
                    whenTouched={() => {
                        cc.app.stopTool('Tool.Annotate.Modify');
                        cc.selectFeature(f, true);
                        cc.app.startTool('Tool.Annotate.Modify');
                    }}
                    content={f.elements.label || '...'}
                />}

                rightButton={f => <gws.components.list.Button
                    className="modAnnotateDeleteListButton"
                    whenTouched={() => cc.removeFeature(f)}
                />}

                withZoom
            />
        } else {
            tabBody = <sidebar.EmptyTabBody>
                {this.__('modAnnotateNotFound')}
            </sidebar.EmptyTabBody>

        }


        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modAnnotateSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {tabBody}
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        {...gws.tools.cls('modAnnotateEditAuxButton', this.props.appActiveTool === 'Tool.Annotate.Modify' && 'isActive')}
                        tooltip={this.__('modAnnotateEditAuxButton')}
                        disabled={features.length === 0}
                        whenTouched={() => cc.app.startTool('Tool.Annotate.Modify')}
                    />
                    <sidebar.AuxButton
                        {...gws.tools.cls('modAnnotateAddAuxButton')}
                        tooltip={this.props.controller.__('modAnnotateAddAuxButton')}
                        whenTouched={() => cc.app.toggleTool('Tool.Annotate.Draw')}
                    />
                    <Cell flex/>
                    {storage.auxButtons(cc, {
                        category: STORAGE_CATEGORY,
                        hasData: features.length > 0,
                        getData: name => cc.storageGetData(name),
                        dataReader: (name, data) => cc.storageReader(name, data)
                    })}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }

}

class AnnotateSidebarView extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);
        let feature = this.props.mapFocusedFeature;

        if (feature && feature.layer !== cc.layer)
            feature = null;


        if (!feature) {
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

class Controller extends gws.Controller {
    uid = MASTER;
    layer: gws.types.IFeatureLayer;
    modifyTool: AnnotateModifyTool;
    selectedStyle: gws.types.IStyle;

    async init() {
        await super.init();

        this.layer = this.map.addServiceLayer(new gws.map.layer.FeatureLayer(this.map, {
            uid: '_annotate',
        }));

        this.layer.cssSelector = '.modAnnotateFeature';

        this.update({
            annotateLastCssSelector: '.modAnnotateFeature',
            annotateLabelTemplates: defaultLabelTemplates,
        });

        this.app.whenChanged('annotateFormData', data => {
            if (data)
                this.updateObject('annotateLabelTemplates', {
                    [data.shapeType]: data.labelTemplate,
                });
        })

        this.app.whenCalled('annotateFromFeature', args => {
            let feature = this.newFromFeature(args.feature);
            if (feature) {
                this.layer.addFeature(feature);
                this.selectFeature(feature, false);
                this.app.startTool('Tool.Annotate.Modify')
            }
        });

        this.app.whenChanged('mapFocusedFeature', feature => {
            if (feature && feature.layer === this.layer) {
                this.update({sidebarActiveTab: 'Sidebar.Annotate'});
                this.app.startTool('Tool.Annotate.Modify');
                this.update({
                    annotateFormData: this.selectedFeature.formData
                })
            } else {
                this.update({
                    annotateFormData: null
                })
            }
            this.layer.features.forEach(f => f.redraw());
        })
    }

    get selectedFeature(): AnnotateFeature | null {
        let ff = this.getValue('mapFocusedFeature');
        if (ff && ff.layer === this.layer)
            return ff;
    }

    get features(): Array<AnnotateFeature> {
        return (this.layer ? this.layer.features : []) as Array<AnnotateFeature>;
    }

    get hasFeatures() {
        return this.features.length > 0;
    }

    //

    newFeatureFromDraw(shapeType, oFeature) {
        let feature = this.newFeature(shapeType, oFeature);
        this.layer.addFeature(feature);
        return feature;

    }

    newStyle(values) {
        let newCssSelector = '.' + gws.tools.uniqId('AnnotateStyle');
        let newStyle = this.app.style.add(new style.Style(newCssSelector, values));
        let focusedStyle = this.map.style.get('.modAnnotateFocus');
        this.map.style.add(
            new style.CascadedStyle(newCssSelector + '.isFocused', [newStyle, focusedStyle]));
        return newCssSelector;
    }

    newFeature(shapeType, oFeature?: ol.Feature) {
        let lastStyle = this.app.style.getFromSelector(this.getValue('annotateLastCssSelector'));
        let newCssSelector = this.newStyle(lastStyle.values);
        let templates = this.getValue('annotateLabelTemplates'),
            labelTemplate = (templates && templates[shapeType]) || defaultLabelTemplates[shapeType];

        let feature = new AnnotateFeature(this.map);
        feature.attributes = {uid: gws.tools.uniqId('annotate')}
        feature.keyName = 'uid';
        feature.geometryName = 'geom';
        feature.cssSelector = newCssSelector;
        feature.shapeType = shapeType;
        feature.labelTemplate = labelTemplate;

        if (oFeature)
            feature.setGeometry(oFeature.getGeometry());

        this.update({
            annotateLastCssSelector: newCssSelector,
            styleEditorCurrentSelector: newCssSelector,
        });

        feature.update();
        feature.redraw();
        return feature;
    }


    //     this.drawFeature = _master(this).newFeature(shapeType, oFeature);


    startLens() {
        let sel = this.getValue('annotateSelectedFeature') as gws.types.IFeature;
        if (sel) {
            this.update({
                lensGeometry: sel.geometry['clone']()
            })
            this.app.startTool('Tool.Lens');
        }

    }

    newFromFeature(f: gws.types.IFeature) {

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


    selectFeature(feature: gws.types.IFeature, panTo) {
        if (panTo) {
            this.update({
                marker: {
                    features: [feature],
                    mode: 'pan',
                }
            });
        }

        this.map.focusFeature(feature);
    }

    removeFeature(f) {
        this.app.stopTool('Tool.Annotate.*');
        this.map.focusFeature(null);
        this.layer.removeFeature(f);
        if (this.hasFeatures)
            this.app.startTool('Tool.Annotate.Modify');
    }

    storageReader(name, data) {
        let lastFeature = null;

        this.app.stopTool('Tool.Annotate.*');
        this.layer.clear();

        let features = [];

        for (let props of data.features) {
            let f = new AnnotateFeature(this.map)
            f.setProps(props);

            if (props.style) {
                f.cssSelector = this.newStyle(props.style.values);
            }

            f.redraw();
            features.push(f);
        }

        this.layer.addFeatures(features);

        if (this.hasFeatures)
            this.app.startTool('Tool.Annotate.Modify');

        // if (lastFeature) {
        //     this.update({
        //         // annotateLastCssClass: lastFeature.styleNames.normal,
        //     });
        // }

    }

    storageGetData(name) {
        return {
            'features': this.features.map(f => f.getProps())
        }
    }

    clear() {
        this.app.stopTool('Tool.Annotate.*');
        this.layer.clear();
    }


}

export const tags = {
    [MASTER]: Controller,
    'Sidebar.Annotate': AnnotateSidebar,
    'Toolbar.Annotate.Draw': AnnotateDrawToolbarButton,
    'Tool.Annotate.Modify': AnnotateModifyTool,
    'Tool.Annotate.Draw': AnnotateDrawTool,

};
