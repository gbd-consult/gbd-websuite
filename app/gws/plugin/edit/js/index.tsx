import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';
import * as toolbar from 'gws/elements/toolbar';
import * as draw from 'gws/elements/draw';
import * as components from 'gws/components';

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;

const MASTER = 'Shared.Edit';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

interface FeatureMap {
    [uid: string]: gws.types.IFeature
}


interface EditState {
    selectedModel?: gws.types.IModel;
    selectedFeature?: gws.types.IFeature;
    formValues?: object;
    formErrors?: object;
    drawFeature?: gws.types.IFeature;
    prevState?: EditState;
    relationFieldName?: string;
    featureMap: { [modelUid: string]: FeatureMap };
    keywordSearchMap: { [modelUid: string]: string };
    suggestSearchMap: { [modelUid: string]: string };
}


type DialogData = {
    type: 'SelectRelation';
    relations: Array<gws.types.IModelRelation>;
    whenRelationSelected: (r: gws.types.IModelRelation) => void;
} | {
    type: 'SelectFeature';
    features: Array<gws.types.IFeature>;
    whenFeatureSelected: (f: gws.types.IFeature) => void;
} | {
    type: 'ConfirmDeleteFeature';
    feature: gws.types.IFeature;
    whenConfirmed: () => void;
} | {
    type: 'Error';
    errorText: string;
    errorDetails: string;
}

interface ViewProps extends gws.types.ViewProps {
    editState: EditState;
    editDialogData?: DialogData;
    editUpdateCount: number;
    // editSearchResults?: SearchResults;
    // editSearchKeywords?: SearchKeywords;
    mapUpdateCount: number;
    appActiveTool: string;
}

const StoreKeys = [
    'editState',
    'editDialogData',
    'editUpdateCount',
    'editSearchText',
    'editSearchResults',
    'editSearchKeywords',
    'mapUpdateCount',
    'appActiveTool',
];


const VALIDATION_ERROR_PREFIX = 'validationError';


const ENABLED_SHAPES_BY_TYPE = {
    'GEOMETRY': null,
    'POINT': ['Point'],
    'LINESTRING': ['Line'],
    'POLYGON': ['Polygon', 'Circle', 'Box'],
    'MULTIPOINT': ['Point'],
    'MULTILINESTRING': ['Line'],
    'MULTIPOLYGON': ['Polygon', 'Circle', 'Box'],
    'GEOMETRYCOLLECTION': null,
};


export class PointerTool extends gws.Tool {
    oFeatureCollection: ol.Collection<ol.Feature>;
    snap: boolean = true;

    async whenPointerDown(evt: ol.MapBrowserEvent) {
        let cc = _master(this);


        if (evt.type !== 'singleclick')
            return

        let currentFeatureClicked = false;

        cc.map.oMap.forEachFeatureAtPixel(evt.pixel, oFeature => {
            if (oFeature === this.oFeatureCollection.item(0)) {
                currentFeatureClicked = true;
            }
        });

        if (currentFeatureClicked) {
            return
        }

        await cc.whenPointerDownAtCoordinate(evt.coordinate);
        let selected = cc.editState.selectedFeature
        if (!selected)
            return

        let oFeature = selected.oFeature
        if (oFeature === this.oFeatureCollection.item(0))
            return

        this.oFeatureCollection.clear()
        this.oFeatureCollection.push(oFeature)
    }

    start() {
        let cc = _master(this);

        this.oFeatureCollection = new ol.Collection<ol.Feature>();

        let ixPointer = new ol.interaction.Pointer({
            handleEvent: evt => this.whenPointerDown(evt)
        });

        let ixModify = this.map.modifyInteraction({
            features: this.oFeatureCollection,
            whenEnded: oFeatures => {
                if (oFeatures[0]) {
                    let feature = oFeatures[0]['_gwsFeature'];
                    if (feature) {
                        cc.whenModifyEnded(feature)


                    }
                }
            }
        });

        let ixs: Array<ol.interaction.Interaction> = [ixPointer, ixModify];

        // if (this.snap && cc.activeLayer)
        //     ixs.push(this.map.snapInteraction({
        //             layer: cc.activeLayer
        //         })
        //     );

        this.map.appendInteractions(ixs);

    }


    stop() {
        this.oFeatureCollection = null;
    }

}


class DrawTool extends draw.Tool {
    whenEnded(shapeType, oFeature) {
        _master(this).whenDrawEnded(oFeature);
    }

    enabledShapes() {
        let model = _master(this).editState.selectedModel;
        if (!model)
            return null;
        return ENABLED_SHAPES_BY_TYPE[model.geometryType.toUpperCase()];
    }

    whenCancelled() {
        _master(this).whenDrawCancelled();
    }
}

interface FeatureListProps extends gws.types.ViewProps {
    features: Array<gws.types.IFeature>;
    whenFeatureTouched: (f: gws.types.IFeature) => void;
    whenSearchChanged: () => void;
}

class FeatureList extends gws.View<FeatureListProps> {
    render() {
        let cc = _master(this);
        let searchText = ''

        let zoomTo = f => this.props.controller.update({
            marker: {
                features: [f],
                mode: 'zoom draw fade'
            }
        });

        let leftButton = f => {
            if (f.geometryName)
                return <components.list.Button
                    className="cmpListZoomListButton"
                    whenTouched={() => zoomTo(f)}
                />
            else
                return <components.list.Button
                    className="cmpListDefaultListButton"
                    whenTouched={() => this.props.whenFeatureTouched(f)}
                />
        }

        return <VBox>
            <VRow>
                <div className="modSearchBox">
                    <Row>
                        <Cell>
                            <gws.ui.Button className='searchIcon'/>
                        </Cell>
                        <Cell flex>
                            <gws.ui.TextInput
                                placeholder={this.__('editSearchPlaceholder')}
                                withClear={true}
                                value={searchText}
                                whenChanged={val => this.props.whenSearchChanged(/*layers, val*/)}
                            />
                        </Cell>
                    </Row>
                </div>
            </VRow>
            <VRow flex>
                <components.feature.List
                    controller={this.props.controller}
                    features={this.props.features}
                    content={f => <gws.ui.Link
                        content={f.views.title}
                        whenTouched={() => this.props.whenFeatureTouched(f)}
                    />}
                    leftButton={leftButton}
                />
            </VRow>
        </VBox>


    }
}

class FeatureDetailsTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);
        let es = this.props.editState;
        let feature = es.selectedFeature;
        let model = es.selectedFeature.model;

        let isDirty = feature.isNew || feature.isDirty;

        let formValues = {
            ...feature.attributes,
            ...feature.editedAttributes,
        };

        let widgets = model.fields.map(f =>
            cc.makeWidget(f, es.selectedFeature, formValues))

        return <sidebar.Tab className="editSidebar editSidebarFormTab">
            <sidebar.TabHeader>
                <Row>
                    <Cell flex>
                        <gws.ui.Title content={feature.views.title}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <VBox>
                    <VRow flex>
                        <Cell flex>
                            <Form tabular>
                                <components.Form
                                    controller={this.props.controller}
                                    feature={feature}
                                    values={formValues}
                                    model={model}
                                    errors={es.formErrors}
                                    widgets={widgets}
                                />
                            </Form>
                        </Cell>
                    </VRow>
                    <VRow>
                        <Row>
                            <Cell flex/>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editSaveButton', isDirty && 'isActive')}
                                    tooltip={this.__('editSave')}
                                    whenTouched={() => cc.whenFormSaveButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editResetButton', isDirty && 'isActive')}
                                    tooltip={this.__('editReset')}
                                    whenTouched={() => cc.whenFormResetButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editCancelButton')}
                                    tooltip={this.__('editCancel')}
                                    whenTouched={() => cc.whenFormCancelButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    className="editDeleteButton"
                                    tooltip={this.__('editDelete')}
                                    whenTouched={() => cc.whenFormDeleteButtonTouched()}
                                />
                            </Cell>
                        </Row>
                    </VRow>
                </VBox>
            </sidebar.TabBody>

        </sidebar.Tab>
    }

}


class FeatureListTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);
        let es = this.props.editState;
        let hasGeom = !gws.lib.isEmpty(es.selectedModel.geometryName);
        let features = Object.values(es.featureMap[es.selectedModel.uid])

        return <sidebar.Tab className="editSidebar">

            <sidebar.TabHeader>
                <Row>
                    <Cell>
                        <gws.ui.Title content={es.selectedModel.title}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <FeatureList
                    controller={cc}
                    whenFeatureTouched={f => cc.whenFeatureListItemTouched(f)}
                    whenSearchChanged={() => []}
                    features={features}
                />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        {...gws.lib.cls('editGotoModelListAuxButton')}
                        tooltip={this.__('editGotoModelListAuxButton')}
                        whenTouched={() => cc.whenGotoModelListButtonTouched()}
                    />
                    <Cell flex/>
                    {es.selectedModel.canCreate && hasGeom && <sidebar.AuxButton
                        {...gws.lib.cls('editDrawAuxButton', this.props.appActiveTool === 'Tool.Edit.Draw' && 'isActive')}
                        tooltip={this.__('editDrawAuxButton')}
                        whenTouched={() => cc.whenDrawButtonTouched()}
                    />}
                    {es.selectedModel.canCreate && <sidebar.AuxButton
                        {...gws.lib.cls('editAddAuxButton')}
                        tooltip={this.__('editAddAuxButton')}
                        whenTouched={() => cc.whenAddButtonTouched()}
                    />}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>

    }
}

class ModelListTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);


        if (gws.lib.isEmpty(cc.editableModels)) {
            return <sidebar.EmptyTab>
                {this.__('editNoLayer')}
            </sidebar.EmptyTab>;
        }

        return <sidebar.Tab className="editSidebar">
            <sidebar.TabHeader>
                <Row>
                    <Cell>
                        <gws.ui.Title content={this.__('editTitle')}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <VBox>
                    <VRow flex>
                        <components.list.List
                            controller={this.props.controller}
                            items={cc.editableModels}
                            content={model => <gws.ui.Link
                                whenTouched={() => cc.whenModelListItemTouched(model)}
                                content={model.title}
                            />}
                            uid={model => model.uid}
                            leftButton={model => <components.list.Button
                                className="editorModelListButton"
                                whenTouched={() => cc.whenModelListItemTouched(model)}
                            />}
                        />
                    </VRow>
                </VBox>
            </sidebar.TabBody>
        </sidebar.Tab>
    }

}

class Dialog extends gws.View<ViewProps> {

    render() {
        let dd = this.props.editDialogData;

        if (!dd || !dd.type) {
            return null;
        }

        let cc = _master(this);

        let cancelButton = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

        if (dd.type === 'SelectRelation') {
            let relations = dd.relations;
            let items = relations.map((r, n) => ({
                value: n,
                text: r.title,
            }));
            let fn = dd.whenRelationSelected;

            return <gws.ui.Dialog
                className="editSelectRelationDialog"
                title={this.__('editSelectRelationTitle')}
                whenClosed={() => cc.closeDialog()}
                buttons={[cancelButton]}
            >
                <Form>
                    <Row>
                        <Cell flex>
                            <gws.ui.List
                                items={items}
                                value={null}
                                whenChanged={v => fn(relations[v])}
                            />
                        </Cell>
                    </Row>
                </Form>
            </gws.ui.Dialog>;
        }


        if (dd.type === 'SelectFeature') {
            let fn = dd.whenFeatureSelected;

            return <gws.ui.Dialog
                className="editSelectFeatureDialog"
                title={this.__('editSelectFeatureTitle')}
                whenClosed={() => cc.closeDialog()}
                buttons={[cancelButton]}
            >
                <Form>
                    <FeatureList
                        controller={cc}
                        whenFeatureTouched={f => fn(f)}
                        whenSearchChanged={() => []}
                        features={dd.features}
                    />
                </Form>
            </gws.ui.Dialog>;
        }

        if (dd.type === 'ConfirmDeleteFeature') {

            let fn = dd.whenConfirmed;

            return <gws.ui.Confirm
                title={this.__('editDeleteFeatureTitle')}
                text={this.__('editDeleteFeatureText').replace(/%s/, dd.feature.views.title)}
                whenConfirmed={() => fn()}
                whenRejected={() => cc.closeDialog()}
            />
        }

        if (dd.type === 'Error') {

            return <gws.ui.Alert
                title={'Fehler'}
                error={dd.errorText}
                details={dd.errorDetails}
                whenClosed={() => cc.closeDialog()}
            />
        }

        return null;

    }
}


class SidebarView extends gws.View<ViewProps> {
    render() {
        let es = this.props.editState;

        if (es.selectedFeature)
            return <FeatureDetailsTab {...this.props} />;

        if (es.selectedModel)
            return <FeatureListTab {...this.props} />;

        return <ModelListTab {...this.props} />;
    }
}

class Sidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'editSidebarIcon';

    get tooltip() {
        return this.__('editSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarView, StoreKeys)
        );
    }
}


class EditToolbarButton extends toolbar.Button {
    iconClass = 'editToolbarButton';
    tool = 'Tool.Edit.Pointer';

    get tooltip() {
        return this.__('editToolbarButton');
    }

}


class EditLayer extends gws.map.layer.FeatureLayer {
    controller: Controller;
    cssSelector = '.editFeature'

    get printPlane() {
        return null;
    }
}


class Controller extends gws.Controller {
    uid = MASTER;
    editableModels: Array<gws.types.IModel>;
    editLayer: EditLayer;

    async init() {
        await super.init();

        this.editableModels = this.app.models.editableModels();
        this.editableModels.sort((a, b) => a.layer.title.localeCompare(b.title))

        let featureMap = {};
        for (let model of this.editableModels)
            featureMap[model.uid] = {}

        this.updateEditState({
            featureMap,
            keywordSearchMap: {},
            suggestSearchMap: {}

        })


        if (!gws.lib.isEmpty(this.editableModels)) {
            this.editLayer = this.map.addServiceLayer(new EditLayer(this.map, {
                uid: '_edit',
            }));
            this.editLayer.controller = this;
        }

        this.app.whenChanged('mapViewState', () => this.whenMapStateChanged());
    }

    //

    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, StoreKeys));
    }

    showDialog(dd: DialogData) {
        this.update({editDialogData: dd})
    }

    closeDialog() {
        this.update({editDialogData: null});
    }

    //


    async whenMapStateChanged() {
        let es = this.editState;

        if (!es.selectedModel || es.selectedFeature)
            return;

        let model = es.selectedModel;

        if (model.loadingStrategy == gws.api.core.FeatureLoadingStrategy.bbox) {
            await this.loadFeaturesForModel(model);
            this.updateEditState({
                selectedModel: model,
            });
        }
    }

    whenModelListItemTouched(model: gws.types.IModel) {
        this.selectModel(model)
    }

    whenFeatureListItemTouched(feature: gws.types.IFeature) {
        this.selectFeature(feature, 'pan');
    }


    async whenPointerDownAtCoordinate(coord: ol.Coordinate) {
        let pt = new ol.geom.Point(coord);

        let res = await this.app.server.editQueryFeatures({
            shapes: [this.map.geom2shape(pt)],
            modelUids: this.editableModels.map(m => m.uid),
            resolution: this.map.viewState.resolution,
        });

        if (gws.lib.isEmpty(res.features)) {
            this.unselectModel()
            return;
        }

        let props = res.features[0];
        let feature = this.app.models.featureFromProps(this.map, props);

        let es = this.editState;
        if (es.selectedModel === feature.model && es.selectedFeature && es.selectedFeature.uid === feature.uid) {
            return
        }

        await this.selectFeature(feature, 'pan');
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Edit'});
    }

    whenModifyEnded(feature: gws.types.IFeature) {
        this.saveFeature(feature, true);

    }

    whenGotoModelListButtonTouched() {
        this.unselectModel()
    }

    async whenAddButtonTouched() {
        let feature = await this.createFeature();
        this.selectFeature(feature, 'pan');
    }

    whenDrawButtonTouched() {
        let es = this.editState;
        this.updateEditState({
            ...es,
            drawFeature: null,
        });
        this.app.startTool('Tool.Edit.Draw')
    }

    async whenDrawEnded(oFeature: ol.Feature) {
        let feature = await this.createFeature(oFeature.getGeometry())
        await this.selectFeature(feature)
        this.app.startTool('Tool.Edit.Pointer');

    }

    whenDrawCancelled() {
    }


    //

    whenFormCancelButtonTouched() {
        this.unselectFeature();
    }

    whenFormDeleteButtonTouched() {
        let feature = this.editState.selectedFeature;

        this.showDialog({
            type: 'ConfirmDeleteFeature',
            feature,
            whenConfirmed: () => {
                this.closeDialog();
                this.deleteFeature(feature);
                this.unselectFeature();
            }
        })
    }

    whenFormResetButtonTouched() {
        let es = this.editState;
        es.selectedFeature.editedAttributes = {};
        this.updateEditState({});
    }

    whenFormSaveButtonTouched() {
        let es = this.editState;
        this.saveFeature(es.selectedFeature);

        this.unselectFeature();
    }

    //

    async whenFormEvent(event: string, widget: gws.types.IModelWidget, field: gws.types.IModelField, value: any) {
        console.log('>> FORM', event, widget, field, value)

        let es = this.editState,
            feature = es.selectedFeature;

        if (!feature)
            return;

        if (event === 'changed') {
            if (es.selectedFeature.getAttribute(field.name) === value) {
                delete feature.editedAttributes[field.name];
            } else {
                feature.editedAttributes[field.name] = value;
            }

            this.updateEditState();
       }
    }


    //

    async selectModel(model: gws.types.IModel) {
        await this.loadFeaturesForModel(model);

        this.updateEditState({
            selectedModel: model,
            selectedFeature: null,
        });
    }

    unselectModel() {
        this.updateEditState({
            selectedModel: null,
            selectedFeature: null,
        })
    }

    async selectFeature(feature: gws.types.IFeature, markerMode = '') {
        await this.loadFeaturesForModel(feature.model, [feature]);

        this.updateEditState({
            selectedModel: feature.model,
            selectedFeature: feature,
        });

        if (markerMode) {
            this.update({
                marker: {
                    features: [feature],
                    mode: markerMode,
                }
            })
        }
    }

    unselectFeature() {
        let f = this.editState.selectedFeature;
        if (f) {
            this.selectModel(f.model)
        }
    }

    async loadFeaturesForModel(model: gws.types.IModel, keepFeatures: Array<gws.types.IFeature> = null) {
        let es = this.editState;

        let queryRequest: gws.api.plugin.edit.action.QueryRequest = {
            modelUids: [model.uid],
            layerUid: model.layer.uid,
            resolution: this.map.viewState.resolution,
        }

        let keyword = es.keywordSearchMap[model.uid];
        if (keyword) {
            queryRequest.keyword = keyword;
        } else if (model.loadingStrategy === gws.api.core.FeatureLoadingStrategy.bbox) {
            queryRequest.extent = this.map.bbox;
        }

        let res = await this.app.server.editQueryFeatures(queryRequest);
        let features = this.app.models.featureListFromProps(this.map, res.features);

        let newMap: FeatureMap = {};

        for (let f of features) {
            newMap[f.uid] = f;
        }

        for (let f of Object.values(es.featureMap[model.uid])) {
            if (f.isDirty) {
                newMap[f.uid] = f;
            }
        }

        if (keepFeatures) {
            for (let f of keepFeatures) {
                newMap[f.uid] = f;
            }
        }

        this.updateEditState({
            featureMap: {
                ...es.featureMap,
                [model.uid]: newMap,
            }
        })
    }

    //

    async serializeValue(val) {
        if (!val)
            return val;

        if (val instanceof FileList) {
            let content: Uint8Array = await gws.lib.readFile(val[0]);
            return {name: val[0].name, content}
        }

        return val;
    }


    async saveFeature(feature: gws.types.IFeature, onlyGeometry = false) {
        let es = this.editState;
        let model = es.selectedModel;
        let featureToWrite = es.selectedFeature.clone();

        let atts = {
            ...feature.attributes,
            ...feature.editedAttributes,
        }

        if (onlyGeometry) {
            atts = {
                [feature.geometryName]: atts[feature.geometryName],
                [feature.keyName]: atts[feature.keyName],
            }
        } else {
            for (let [k, v] of gws.lib.entries(atts)) {
                atts[k] = await this.serializeValue(v);
            }
        }

        featureToWrite.attributes = atts;

        let VALIDATION_ERROR_PREFIX = '-'

        let res = await this.app.server.editWriteFeature({
            modelUid: model.uid,
            layerUid: model.layer.uid,
            feature: featureToWrite.getProps(onlyGeometry ? 0 : 1)
        }, {binary: false});

        if (res.error) {
            this.showDialog({
                type: 'Error',
                errorText: this.__('editSaveErrorText'),
                errorDetails: res.error.info,
            })
            return;
        }

        let newState: Partial<EditState> = {
            formErrors: {},
        };

        let props = res.feature;

        for (let e of props.errors) {
            let msg = e['message'];
            if (msg.startsWith(VALIDATION_ERROR_PREFIX))
                msg = this.__(msg)
            newState.formErrors[e['fieldName']] = msg;
        }

        if (!gws.lib.isEmpty(newState.formErrors)) {
            this.updateEditState(newState);
            return;
        }

        es.selectedFeature.setProps(props)
        this.map.forceUpdate();
    }

    async createFeature(geometry?: ol.geom.Geometry): Promise<gws.types.IFeature> {
        let model = this.editState.selectedModel;
        let attributes = {}

        if (geometry) {
            attributes[model.geometryName] = this.map.geom2shape(geometry)
        }

        let featureToInit = model.featureWithAttributes(this.map, attributes);
        featureToInit.isNew = true;

        let res = await this.app.server.editInitFeature({
            modelUid: model.uid,
            layerUid: model.layer.uid,
            feature: featureToInit.getProps(1)
        });

        let newFeature = model.featureFromProps(this.map, res.feature);
        newFeature.isNew = true;

        this.registerFeature(newFeature);

        return newFeature;
    }

    async deleteFeature(feature: gws.types.IFeature) {

        let res = await this.app.server.editDeleteFeature({
            modelUid: feature.model.uid,
            feature: feature.getProps(0)
        });

        if (res.error) {
            this.showDialog({
                type: 'Error',
                errorText: this.__('editDeleteErrorText'),
                errorDetails: res.error.info,
            })
            return;
        }

        this.map.forceUpdate();

    }

    registerFeature(feature) {
        let fm = this.editState.featureMap[feature.modelUid] || {};
        this.updateEditState({
            featureMap: {
                ...this.editState.featureMap,
                [feature.modelUid]: {
                    ...fm,
                    [feature.uid]: feature,
                }
            }
        })
    }


    //


    makeWidget(field: gws.types.IModelField, feature: gws.types.IFeature, values: gws.types.Dict): React.ReactElement | null {
        let p = field.widgetProps;

        if (!p)
            return null;

        let cls = this.app.getClass('ModelWidget.' + p.type);

        let props = {
            controller: this,
            feature,
            field,
            values,
            options: p.options,
            readOnly: p.readOnly,
            when: (event, widget, field, value) => this.whenFormEvent(event, widget, field, value)
        }

        return React.createElement(cls, props);

    }

    //


    get editState(): EditState {
        return this.getValue('editState') || {};
    }

    updateEditState(es: Partial<EditState> = null) {
        let editState = {
            ...this.editState,
            ...(es || {})
        }

        this.update({editState})

        if (this.editLayer) {

            this.editLayer.clear();
            let f = editState.selectedFeature;

            if (f) {
                this.editLayer.addFeature(f);
                this.app.map.focusFeature(f);
                f.redraw()
            }
        }


    }


}

gws.registerTags({
    'Shared.Edit': Controller,
    'Sidebar.Edit': Sidebar,
    'Tool.Edit.Pointer': PointerTool,
    'Tool.Edit.Draw': DrawTool,
    'Toolbar.Edit': EditToolbarButton,
});
