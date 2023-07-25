import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as measure from 'gws/map/measure';
import * as template from 'gws/map/template';
import * as style from 'gws/map/style';
import * as styler from 'gws/elements/styler';
import * as draw from 'gws/elements/draw';
import * as modify from 'gws/elements/modify';
import * as storage from 'gws/elements/storage';
import * as sidebar from 'gws/elements/sidebar';
import * as toolbar from 'gws/elements/toolbar';
import * as components from 'gws/components';


const MASTER = 'Shared.Annotate';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as Controller;

let {Form, Row, Cell} = gws.ui.Layout;

const defaultLabelTemplates = {
    Point: '{xy}',
    Line: '{len}',
    Polygon: '{area}',
    Circle: '{radius}',
    Box: '{width} x {height}',
};

interface FormData {
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
    annotateCurrentStyle: gws.types.IStyle;
    annotateFeatureCount: number;
    annotateFeatures: Array<Feature>;
    annotateFormData: FormData;
    annotateLabelTemplates: gws.types.Dict;
    annotateSelectedFeature: Feature;
    annotateTab: string;
    appActiveTool: string;
}

const StoreKeys = [
    'annotateCurrentStyle',
    'annotateFeatureCount',
    'annotateFeatures',
    'annotateFormData',
    'annotateLabelTemplates',
    'annotateSelectedFeature',
    'annotateTab',
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

class Feature extends gws.map.Feature {
    get formData(): FormData {
        return {
            shapeType: this.attributes.shapeType,
            labelTemplate: this.attributes.labelTemplate,
            x: this.format('{x|M|2}'),
            y: this.format('{y|M|2}'),
            radius: this.format('{radius|M|2}'),
            width: this.format('{width|M|2}'),
            height: this.format('{height|M|2}'),
        }
    }

    get master(): Controller {
        return this.map.app.controller(MASTER) as Controller;
    }

    redraw() {
        this.views['label'] = this.format(this.attributes.labelTemplate);
        return super.redraw();
    }

    updateFromForm(ff: FormData) {
        this.attributes.labelTemplate = ff.labelTemplate;

        let t = this.attributes.shapeType;

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

        this.redraw();
    }

    protected format(text) {
        return template.formatGeometry(text, this.geometry, this.map.projection);
    }

}


class Layer extends gws.map.layer.FeatureLayer {
}

class DrawTool extends draw.Tool {
    styleName = '.annotateFeature';

    get title() {
        return this.__('annotateDrawToolbarButton')
    }

    start() {
        super.start();
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Annotate'});
    }

    whenStarted(shapeType, oFeature) {
        _master(this).whenDrawStarted(shapeType, oFeature)
    }

    whenEnded(shapeType, oFeature) {
        _master(this).whenDrawEnded(shapeType, oFeature)
    }

    whenCancelled() {
        _master(this).whenDrawCancelled();

    }

}

class ModifyTool extends modify.Tool {
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
        _master(this).whenFeatureSelectedInMap(f);
    }

    whenUnselected() {
        _master(this).whenFeatureUnselectedInMap()
    }
}

class FeatureForm extends gws.View<ViewProps> {

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
            st = selectedFeature.attributes.shapeType;

        // let submit = () => {
        //     selectedFeature.updateFromForm(this.props.annotateFormData);
        //     cc.unselectFeature();
        //     cc.app.stopTool('Tool.Annotate.Modify');
        // };

        let bind = (prop) => ({
            value: this.props.annotateFormData[prop],
            whenChanged: val => cc.whenFeaturePropertyChanged(selectedFeature, prop, val),
        });

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

            cc.whenFeaturePropertyChanged(selectedFeature, 'labelTemplate', val)
        };

        let form = [];

        if (['Point', 'Box', 'Circle'].includes(st)) {
            form.push(<gws.ui.NumberInput
                step={1}
                locale={this.props.controller.app.locale}
                label={this.__('annotateX')}
                {...bind('x')}
            />);
            form.push(<gws.ui.NumberInput
                step={1}
                locale={this.props.controller.app.locale}
                label={this.__('annotateY')}
                {...bind('y')}
            />);
        }

        if (st === 'Box') {
            form.push(<gws.ui.NumberInput
                step={1}
                locale={this.props.controller.app.locale}
                label={this.__('annotateWidth')}
                {...bind('width')}
            />)
            form.push(<gws.ui.NumberInput
                step={1}
                locale={this.props.controller.app.locale}
                label={this.__('annotateHeight')}
                {...bind('height')}
            />)
        }

        if (st === 'Circle') {
            form.push(<gws.ui.NumberInput
                step={1}
                locale={this.props.controller.app.locale}
                label={this.__('annotateRadius')}
                {...bind('radius')}
            />);
        }

        form.push(<gws.ui.TextArea
            focusRef={labelEditorRef}
            label={this.__('annotateLabelEdit')}
            {...bind('labelTemplate')}
        />);

        form.push(<gws.ui.Select
            label={this.__('annotatePlaceholder')}
            value=''
            items={placeholders}
            whenChanged={insertPlacehodler}
        />);

        return <div className="annotateFeatureDetails">
            <Form tabular children={form}/>
            <Form>
                <Row>
                    <Cell flex/>
                    {/*<Cell>*/}
                    {/*    <gws.ui.Button*/}
                    {/*        className="cmpButtonFormOk"*/}
                    {/*        tooltip={this.props.controller.__('annotateSaveButton')}*/}
                    {/*        whenTouched={submit}*/}
                    {/*    />*/}
                    {/*</Cell>*/}
                    <Cell spaced>
                        <gws.ui.Button
                            className="annotateStyleButton"
                            tooltip={this.__('annotateStyleButton')}
                            whenTouched={() => cc.update({annotateTab: 'style'})}
                        />
                    </Cell>
                    <Cell spaced>
                        <gws.ui.Button
                            {...gws.lib.cls('annotateCancelButton')}
                            tooltip={this.__('annotateCancelButton')}
                            whenTouched={() => cc.whenFeatureFormClosed()}
                        />
                    </Cell>
                    <Cell spaced>
                        <gws.ui.Button
                            className="annotateRemoveButton"
                            tooltip={this.__('annotateRemoveButton')}
                            whenTouched={() => cc.whenFeatureRemoveButtonTouched(selectedFeature)}
                        />
                    </Cell>
                </Row>
            </Form>
        </div>
    }
}

class FeatureTabFooter extends gws.View<ViewProps> {
    render() {
        let cc = _master(this.props.controller),
            selectedFeature = this.props.annotateSelectedFeature,
            tab = this.props.annotateTab === 'style' ? 'style' : 'form';

        return <sidebar.TabFooter>
            <sidebar.AuxToolbar>
                <Cell>
                    <sidebar.AuxButton
                        {...gws.lib.cls('annotateFormAuxButton', tab === 'form' && 'isActive')}
                        whenTouched={() => cc.update({annotateTab: 'form'})}
                        tooltip={cc.__('annotateFormAuxButton')}
                    />
                </Cell>
                <Cell>
                    <sidebar.AuxButton
                        {...gws.lib.cls('annotateStyleAuxButton', tab === 'style' && 'isActive')}
                        whenTouched={() => cc.update({annotateTab: 'style'})}
                        tooltip={cc.__('annotateStyleAuxButton')}
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
            </sidebar.AuxToolbar>
        </sidebar.TabFooter>
    }
}

class FeatureFormTab extends gws.View<ViewProps> {
    render() {
        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('annotateSidebarDetailsTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <FeatureForm {...this.props} />
            </sidebar.TabBody>

            <FeatureTabFooter {...this.props}/>
        </sidebar.Tab>
    }
}

class FeatureStyleTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this.props.controller),
            sc = cc.app.controller('Shared.Style') as styler.Controller;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('annotateSidebarDetailsTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {sc.form()}
            </sidebar.TabBody>

            <FeatureTabFooter {...this.props}/>
        </sidebar.Tab>
    }

}

class ListTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this.props.controller),
            selectedFeature = this.props.annotateSelectedFeature,
            hasFeatures = this.props.annotateFeatures && this.props.annotateFeatures.length > 0;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('annotateSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {cc.hasFeatures
                    ? <components.feature.List
                        controller={cc}
                        features={this.props.annotateFeatures || []}
                        isSelected={f => f === selectedFeature}

                        content={(f: Feature) => <gws.ui.Link
                            whenTouched={() => cc.whenFeatureNameTouched(f)}
                            content={f.views.label || '...'}
                        />}

                        rightButton={(f: Feature) => <components.list.Button
                            className="annotateDeleteListButton"
                            whenTouched={() => cc.whenFeatureRemoveButtonTouched(f)}
                        />}

                        withZoom
                    />
                    : <sidebar.EmptyTabBody>
                        {this.__('annotateNotFound')}
                    </sidebar.EmptyTabBody>
                }
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        {...gws.lib.cls('annotateEditAuxButton', this.props.appActiveTool === 'Tool.Annotate.Modify' && 'isActive')}
                        tooltip={this.__('annotateEditAuxButton')}
                        disabled={!hasFeatures}
                        whenTouched={() => cc.app.startTool('Tool.Annotate.Modify')}
                    />
                    <sidebar.AuxButton
                        {...gws.lib.cls('annotateAddAuxButton')}
                        tooltip={this.props.controller.__('annotateAddAuxButton')}
                        whenTouched={() => cc.app.toggleTool('Tool.Annotate.Draw')}
                    />
                    <Cell flex/>
                    <storage.AuxButtons
                        controller={cc}
                        actionName={'annotateStorage'}
                        hasData={hasFeatures}
                        getData={() => cc.storageGetData()}
                        loadData={(data) => cc.storageLoadData(data)}
                    />
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }

}

class SidebarView extends gws.View<ViewProps> {
    render() {
        if (!this.props.annotateSelectedFeature) {
            return <ListTab {...this.props}/>;
        }
        if (this.props.annotateTab === 'style') {
            return <FeatureStyleTab {...this.props}/>;
        }
        return <FeatureFormTab {...this.props}/>;
    }
}

class Sidebar extends gws.Controller implements gws.types.ISidebarItem {

    iconClass = 'annotateSidebarIcon';

    get tooltip() {
        return this.__('annotateSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarView, StoreKeys)
        );
    }

}

class DrawToolbarButton extends toolbar.Button {
    iconClass = 'annotateDrawToolbarButton';
    tool = 'Tool.Annotate.Draw';

    get tooltip() {
        return this.__('annotateDrawToolbarButton');
    }
}

class Controller extends gws.Controller {
    uid = MASTER;
    layer: Layer;
    drawFeature?: Feature = null;

    async init() {
        await super.init();

        this.layer = this.map.addServiceLayer(new Layer(this.map, {
            uid: '_annotate',
        }));

        this.update({
            annotateCurrentStyle: this.map.style.get('.annotateFeature'),
            annotateLabelTemplates: defaultLabelTemplates,
        });

        let props = this.app.actionProps('annotate') as gws.api.plugin.annotate.Props;
        if (props) {
            this.updateObject('storageState', {
                annotateStorage: props.storage ? props.storage.state : null,
            })
        }

        this.app.whenCalled('annotateFromFeature', args =>
            this.createNewFeatureFromFeature(args.feature)
        );
    }

    get features(): Array<Feature> {
        return (this.layer ? this.layer.features : []) as Array<Feature>;
    }

    get hasFeatures() {
        return this.features.length > 0;
    }


    //

    whenDrawStarted(shapeType, oFeature) {
        this.unselectFeature();
        this.drawFeature = this.newFeatureOfType(shapeType);
        this.drawFeature.setOlFeature(oFeature);
        this.drawFeature.setStyle(this.getValue('annotateCurrentStyle'));
        this.addFeature(this.drawFeature);
    }

    whenDrawEnded(shapeType, oFeature) {
        this.layer.removeFeature(this.drawFeature);
        this.drawFeature = null;
        this.createNewFeature(shapeType, oFeature.getGeometry());
    }

    whenDrawCancelled() {
        if (this.drawFeature) {
            this.layer.removeFeature(this.drawFeature);
            this.drawFeature = null;
        }
    }

    whenFeatureNameTouched(f: Feature) {
        this.app.stopTool('Tool.Annotate.Modify');
        this.selectFeature(f);
        this.panToFeature(f);
        this.app.startTool('Tool.Annotate.Modify');
    }

    whenFeatureRemoveButtonTouched(f: Feature) {
        this.removeFeature(f);
    }

    whenFeatureSelectedInMap(f: Feature) {
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Annotate'});
        this.selectFeature(f);
    }

    whenFeatureUnselectedInMap() {
        this.unselectFeature();
    }

    whenFeaturePropertyChanged(f: Feature, prop, val) {
        let data = this.getValue('annotateFormData') || {};
        data = {...data, [prop]: val};
        this.update({annotateFormData: data});
        f.updateFromForm(data);
        this.updateObject('annotateLabelTemplates', {
            [f.attributes.shapeType]: data.labelTemplate,
        });
    }

    whenFeatureFormClosed() {
        this.unselectFeature();
        this.app.stopTool('Tool.Annotate.Modify');
    }


    //

    createNewFeature(shapeType, geom) {
        let f = this.newFeatureOfType(shapeType);

        f.setGeometry(geom);

        f.setStyle(this.cloneStyle(this.getValue('annotateCurrentStyle')));

        this.addFeature(f);
        // this.selectFeature(f);
    }

    createNewFeatureFromFeature(src: gws.types.IFeature) {
        if (!src || !src.geometry)
            return;

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

        let geom = singleGeometry(src.geometry);
        if (!geom)
            return;

        let shapeType = geom.getType();
        if (shapeType === 'LineString')
            shapeType = 'Line';

        return this.createNewFeature(shapeType, geom);
    }

    newFeatureOfType(shapeType): Feature {
        let templates = this.getValue('annotateLabelTemplates'),
            labelTemplate = (templates && templates[shapeType]) || defaultLabelTemplates[shapeType];

        let f = new Feature(this.app.models.defaultModel());

        f.setAttributes({
            uid: gws.lib.uniqId('annotate'),
            shapeType,
            labelTemplate,
        });

        return f;
    }

    cloneStyle(currStyle) {
        let uid = gws.lib.uniqId('annotateStyle');
        let newStyle = this.map.style.copy(currStyle, '.' + uid);
        this.createFocusStyle(newStyle);
        return newStyle;
    }

    createFocusStyle(sty) {
        let focusedStyle = this.map.style.get('.annotateFocused');
        this.map.style.add(
            new style.CascadedStyle(sty.cssSelector + '.isFocused', [sty, focusedStyle]));
    }

    panToFeature(f: Feature) {
        this.update({
            marker: {
                features: [f],
                mode: 'pan',
            }
        });
    }

    selectFeature(f: Feature) {
        this.features.forEach(f => f.setSelected(false));
        this.map.focusFeature(f);

        let sty = this.map.style.at(f.cssSelector);

        this.update({
            annotateSelectedFeature: f,
            annotateFormData: f.formData,
            annotateCurrentStyle: sty,
            stylerCurrentStyle: sty,
        });
    }

    unselectFeature() {
        this.features.forEach(f => f.setSelected(false));
        this.map.focusFeature(null);
        this.update({
            annotateSelectedFeature: null,
            annotateFormData: {},
        });
    }

    addFeature(f) {
        this.layer.addFeature(f);
        this.update({
            annotateFeatures: this.layer.features,
        });
    }

    removeFeature(f) {
        this.app.stopTool('Tool.Annotate.*');
        this.unselectFeature();
        this.layer.removeFeature(f);
        if (this.hasFeatures)
            this.app.startTool('Tool.Annotate.Modify');
        this.update({
            annotateFeatures: this.layer.features,
        });
    }

    clear() {
        this.app.stopTool('Tool.Annotate.*');
        this.layer.clear();
        this.update({
            annotateFeatures: this.layer.features,
        });
    }

    storageLoadData(data) {
        let lastFeature = null;

        this.app.stopTool('Tool.Annotate.*');

        // @TODO an option to add data
        this.clear();

        let sty;
        for (let p of (data.styles || [])) {
            let s = p.cssSelector;
            if (s && s.includes('annotateStyle')) {
                sty = this.map.style.loadFromProps(p);
                this.createFocusStyle(sty);
            }
        }

        for (let p of data.features) {
            let f = new Feature(this.app.models.defaultModel());
            f.setProps(p);
            this.layer.addFeature(f);
        }

        this.update({
            annotateFeatures: this.layer.features,
        });

        if (sty) {
            this.update({annotateCurrentStyle: sty})
        }

    }

    storageGetData() {
        return {
            'features': this.features.map(f => f.getProps()),
            'styles': this.map.style.props,
        }
    }


}

gws.registerTags({
    [MASTER]: Controller,
    'Sidebar.Annotate': Sidebar,
    'Toolbar.Annotate.Draw': DrawToolbarButton,
    'Tool.Annotate.Modify': ModifyTool,
    'Tool.Annotate.Draw': DrawTool,
});
      