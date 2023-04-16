import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';
import * as toolbar from 'gws/elements/toolbar';
import * as draw from 'gws/elements/draw';
import * as components from 'gws/components';

// see app/gws/base/model/validator.py
const DEFAULT_MESSAGE_PREFIX = 'validationError_';


let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;

const MASTER = 'Shared.Edit';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

interface FeatureCacheElement {
    keyword: string;
    extent: string;
    features: Array<gws.types.IFeature>,
}

interface EditState {
    selectedModel?: gws.types.IModel;
    selectedFeature?: gws.types.IFeature;
    formValues?: object;
    formErrors?: object;
    drawFeature?: gws.types.IFeature;
    prevState?: EditState;
    relationFieldName?: string;
    searchText: { [modelUid: string]: string };
    featureCache: { [key: string]: FeatureCacheElement },
}

interface SelectRelationDialogData {
    type: 'SelectRelation';
    relations: Array<gws.types.IModelRelation>;
    whenRelationSelected: (r: gws.types.IModelRelation) => void;
}

interface SelectFeatureDialogData {
    type: 'SelectFeature';
    field: gws.types.IModelField;
    features: Array<gws.types.IFeature>;
    whenFeatureTouched: (f: gws.types.IFeature) => void;
    whenSearchTextChanged: (val: string) => void;
}

interface DeleteFeatureDialogData {
    type: 'DeleteFeature';
    feature: gws.types.IFeature;
    whenConfirmed: () => void;
}

interface ErrorDialogData {
    type: 'Error';
    errorText: string;
    errorDetails: string;
}

type DialogData =
    SelectRelationDialogData
    | SelectFeatureDialogData
    | DeleteFeatureDialogData
    | ErrorDialogData;


interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    editState: EditState;
    editDialogData?: DialogData;
    appActiveTool: string;
}

const StoreKeys = [
    'editState',
    'editDialogData',
    'appActiveTool',
];


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

    async init() {
        await super.init();
        this.app.whenChanged('mapFocusedFeature', () => {
            this.setFeature(this.getValue('mapFocusedFeature'));
        })
    }

    setFeature(feature?: gws.types.IFeature) {
        if (!this.oFeatureCollection)
            this.oFeatureCollection = new ol.Collection<ol.Feature>();

        this.oFeatureCollection.clear();

        if (!feature || !feature.geometryName) {
            return;
        }

        let oFeature = feature.oFeature
        if (oFeature === this.oFeatureCollection.item(0)) {
            return;
        }

        this.oFeatureCollection.push(oFeature)
    }

    async whenPointerDown(evt: ol.MapBrowserEvent) {
        let cc = _master(this);

        if (!this.oFeatureCollection)
            this.oFeatureCollection = new ol.Collection<ol.Feature>();

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
    }

    start() {
        let cc = _master(this);

        this.setFeature(cc.editState.selectedFeature);

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


class FeatureDetailsTab extends gws.View<ViewProps> {
    async componentDidUpdate(prevProps) {
        let cc = _master(this);
        await cc.prepareFeatureForm();
    }


    render() {
        let cc = _master(this);
        let es = this.props.editState;
        let feature = es.selectedFeature;
        let model = es.selectedFeature.model;

        let values = feature.currentAttributes();

        let widgets = [];
        for (let f of feature.model.fields)
            widgets.push(cc.createWidget(f, feature, values));

        let isDirty = feature.isNew || feature.isDirty;

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
                            <Form>
                                <components.Form
                                    controller={this.props.controller}
                                    feature={feature}
                                    values={values}
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

interface FeatureListProps extends gws.types.ViewProps {
    features: Array<gws.types.IFeature>;
    whenFeatureTouched: (f: gws.types.IFeature) => void;
    withSearch: boolean;
    whenSearchTextChanged: (val: string) => void;
    searchText: string;
}

class FeatureList extends gws.View<FeatureListProps> {
    render() {
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
            {this.props.withSearch && <VRow>
                <div className="modSearchBox">
                    <Row>
                        <Cell>
                            <gws.ui.Button className='searchIcon'/>
                        </Cell>
                        <Cell flex>
                            <gws.ui.TextInput
                                placeholder={this.__('editSearchPlaceholder')}
                                withClear={true}
                                value={this.props.searchText}
                                whenChanged={val => this.props.whenSearchTextChanged(val)}
                            />
                        </Cell>
                    </Row>
                </div>
            </VRow>}
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

class FeatureListTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);
        let es = this.props.editState;
        let model = es.selectedModel;
        let hasGeom = !gws.lib.isEmpty(model.geometryName);
        let features = cc.loadedFeatures(model.uid);
        let searchText = es.searchText[model.uid] || '';

        return <sidebar.Tab className="editSidebar">

            <sidebar.TabHeader>
                <Row>
                    <Cell>
                        <gws.ui.Title content={model.title}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <FeatureList
                    controller={cc}
                    whenFeatureTouched={f => cc.whenFeatureListItemTouched(f)}
                    whenSearchTextChanged={val => cc.whenListSearchTextChanged(val)}
                    features={features}
                    searchText={searchText}
                    withSearch={model.supportsKeywordSearch}
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
                    {model.canCreate && hasGeom && <sidebar.AuxButton
                        {...gws.lib.cls('editDrawAuxButton', this.props.appActiveTool === 'Tool.Edit.Draw' && 'isActive')}
                        tooltip={this.__('editDrawAuxButton')}
                        whenTouched={() => cc.whenDrawButtonTouched()}
                    />}
                    {model.canCreate && <sidebar.AuxButton
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
        let items = [...cc.models];

        items.sort((a, b) => a.layer.title.localeCompare(b.title));

        if (gws.lib.isEmpty(cc.models)) {
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
                            items={items}
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

        switch (dd.type) {
            case 'SelectRelation':
                return <SelectRelationDialog {...this.props} />
            case 'SelectFeature':
                return <SelectFeatureDialog {...this.props} />
            case 'DeleteFeature':
                return <DeleteFeatureDialog {...this.props} />
            case 'Error':
                return <ErrorDialog {...this.props} />
        }
    }
}

class SelectRelationDialog extends gws.View<ViewProps> {
    render() {
        let dd = this.props.editDialogData as SelectRelationDialogData;
        let cc = _master(this);

        let cancelButton = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

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
}

class SelectFeatureDialog extends gws.View<ViewProps> {
    render() {
        let cc = _master(this);
        let dd = this.props.editDialogData as SelectFeatureDialogData;
        let searchText = this.props.editState.searchText[dd.field.uid] || '';
        let features = cc.loadedFeatures(dd.field.uid);

        let cancelButton = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

        let relModel = cc.relatedModelForField(dd.field)

        return <gws.ui.Dialog
            className="editSelectFeatureDialog"
            title={this.__('editSelectFeatureTitle')}
            whenClosed={() => cc.closeDialog()}
            buttons={[cancelButton]}
        >
            <FeatureList
                controller={cc}
                whenFeatureTouched={dd.whenFeatureTouched}
                whenSearchTextChanged={dd.whenSearchTextChanged}
                features={features}
                searchText={searchText}
                withSearch={relModel && relModel.supportsKeywordSearch}
            />
        </gws.ui.Dialog>;
    }
}

class DeleteFeatureDialog extends gws.View<ViewProps> {
    render() {
        let dd = this.props.editDialogData as DeleteFeatureDialogData;
        let cc = _master(this);

        return <gws.ui.Confirm
            title={this.__('editDeleteFeatureTitle')}
            text={this.__('editDeleteFeatureText').replace(/%s/, dd.feature.views.title)}
            whenConfirmed={() => dd.whenConfirmed()}
            whenRejected={() => cc.closeDialog()}
        />
    }
}

class ErrorDialog extends gws.View<ViewProps> {
    render() {
        let dd = this.props.editDialogData as ErrorDialogData;
        let cc = _master(this);

        return <gws.ui.Alert
            title={'Fehler'}
            error={dd.errorText}
            details={dd.errorDetails}
            whenClosed={() => cc.closeDialog()}
        />
    }
}

//

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
    models: Array<gws.types.IModel>;
    editLayer: EditLayer;

    async init() {
        await super.init();

        this.models = this.app.models.editableModels();

        this.updateEditState({
            searchText: {},
            featureCache: {},
        });


        if (!gws.lib.isEmpty(this.models)) {
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
        let model = es.selectedModel;

        if (!model)
            return;

        if (model.loadingStrategy == gws.api.core.FeatureLoadingStrategy.bbox) {
            await this.loadFeaturesForList(model);
        }
    }

    async whenModelListItemTouched(model: gws.types.IModel) {
        await this.selectModel(model)
    }

    async whenFeatureListItemTouched(feature: gws.types.IFeature) {
        let res = await this.app.server.editQueryFeatures({
            modelUids: this.models.map(m => m.uid),
            featureUids: [feature.uid],
            resolution: this.map.viewState.resolution,
            relationDepth: 1,
        });

        if (gws.lib.isEmpty(res.features)) {
            return;
        }

        let loaded = this.app.models.featureFromProps(res.features[0]);

        await this.selectFeature(loaded, 'pan');
    }

    async whenListSearchTextChanged(val: string) {
        let model = this.editState.selectedModel;
        await this.updateSearchText(model.uid, val);
        await this.loadFeaturesForList(model);
    }

    async whenPointerDownAtCoordinate(coord: ol.Coordinate) {
        let pt = new ol.geom.Point(coord);
        let models;

        // @TODO should be an option whether to query the selected model only or all of them
        if (this.editState.selectedModel)
            models = [this.editState.selectedModel]
        else
            models = this.models.filter(m => m.layer ? m.layer.visible : true);

        let res = await this.app.server.editQueryFeatures({
            shapes: [this.map.geom2shape(pt)],
            modelUids: models.map(m => m.uid),
            resolution: this.map.viewState.resolution,
            relationDepth: 1,
        });

        if (gws.lib.isEmpty(res.features)) {
            this.unselectFeature();
            return;
        }

        let loaded = this.app.models.featureFromProps(res.features[0]);
        let sel = this.editState.selectedFeature;

        if (sel && sel.model === loaded.model && sel.uid === loaded.uid) {
            return;
        }

        await this.selectFeature(loaded);
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Edit'});
    }

    _geometryTimer = 0

    async whenModifyEnded(feature: gws.types.IFeature) {
        console.log('edit: whenModifyEnded')
        let save = async () => {
            console.log('edit: save geometry')
            let ok = await this.saveFeature(feature, true);
            if (ok) {
                feature.commitEdits();
                this.map.forceUpdate();
            }
        }
        clearTimeout(this._geometryTimer);
        this._geometryTimer = Number(setTimeout(save, 1000));
    }

    whenGotoModelListButtonTouched() {
        this.unselectModel();
    }

    async whenAddButtonTouched() {
        let feature = await this.createFeature();
        await this.selectFeature(feature, 'pan');
    }

    whenDrawButtonTouched() {
        this.updateEditState({
            drawFeature: null,
        });
        this.app.startTool('Tool.Edit.Draw')
    }

    async whenDrawEnded(oFeature: ol.Feature) {
        let feature = await this.createFeature(oFeature.getGeometry())
        await this.selectFeature(feature)
    }

    whenDrawCancelled() {
    }


    //

    async whenFormCancelButtonTouched() {
        let sf = this.editState.selectedFeature;
        await this.unselectFeature();
        await this.selectModel(sf.model);
    }

    whenFormDeleteButtonTouched() {
        let sf = this.editState.selectedFeature;
        let t = this;

        async function confirmed() {
            t.closeDialog();
            await t.deleteFeature(sf);
            await t.unselectFeature();
            await t.selectModel(sf.model);
        }

        this.showDialog({
            type: 'DeleteFeature',
            feature: sf,
            whenConfirmed: confirmed
        })
    }

    whenFormResetButtonTouched() {
        let sf = this.editState.selectedFeature;
        sf.resetEdits();
        this.updateEditState();
    }

    async whenFormSaveButtonTouched() {
        let es = this.editState;
        let feature = es.selectedFeature;

        let ok = await this.saveFeature(feature);
        if (ok) {
            feature.commitEdits();
            this.map.forceUpdate();

            await this.loadFeaturesForList(feature.model, true);
            this.updateEditState({
                selectedFeature: null,
                selectedModel: feature.model,
            });
        }

    }

    //

    async selectModel(model: gws.types.IModel) {
        await this.loadFeaturesForList(model);
        this.updateEditState({
            selectedModel: model,
        });
    }

    unselectModel() {
        this.updateEditState({
            selectedModel: null,
        })
    }

    async selectFeature(feature: gws.types.IFeature, markerMode = '') {
        this.updateEditState({
            selectedFeature: feature,
            formErrors: null,
        });

        if (markerMode) {
            this.update({
                marker: {
                    features: [feature],
                    mode: markerMode,
                }
            })
        }

        this.app.startTool('Tool.Edit.Pointer');
    }

    unselectFeature() {
        this.updateEditState({
            selectedFeature: null
        })
    }

    async prepareFeatureForm() {
        let sf = this.editState.selectedFeature;

        if (!sf)
            return;

        for (let f of sf.model.fields)
            await this.prepareWidget(f, sf);
    }

    async prepareWidget(field: gws.types.IModelField, feature: gws.types.IFeature) {
        let p = field.widgetProps;

        if (!p)
            return null;

        if (p.type === 'featureSelect') {
            await this.loadRelatedFeaturesForField(field)
        }

        if (p.type === 'featureSuggest') {
            await this.loadRelatedFeaturesForField(field)
        }

    }

    createWidget(field: gws.types.IModelField, feature: gws.types.IFeature, values: gws.types.Dict): React.ReactElement | null {
        let p = field.widgetProps;

        if (!p)
            return null;

        let tag = 'ModelWidget.' + p.type;
        let controller = (this.app.controllerByTag(tag) || this.app.createControllerFromConfig(this, {tag})) as gws.types.IModelWidget;

        let props: gws.types.Dict = {
            controller,
            feature,
            field,
            widgetProps: field.widgetProps,
            values,
            whenChanged: val => this.whenWidgetChanged(field, val),
            whenEntered: val => this.whenWidgetEntered(field, val),
        }

        if (p.type === 'geometry') {
            props.whenNewButtonTouched = () => this.whenGeometryWidgetNewButtonTouched(field);
            props.whenEditButtonTouched = () => this.whenGeometryWidgetEditButtonTouched(field);
        }

        if (p.type == 'featureSelect') {
            let features = this.loadedFeatures(field.uid);
            if (!features)
                return null;
            props.features = features;
        }

        if (p.type == 'featureSuggest') {
            let features = this.loadedFeatures(field.uid);
            if (!features)
                return null;
            props.features = features;
            props.searchText = this.editState.searchText[field.uid] || '';
            props.whenSearchTextChanged = val => this.updateSearchText(field.uid, val);
        }

        if (p.type === 'featureList') {
            props.whenNewButtonTouched = () => this.whenFeatureListWidgetNewButtonTouched(field);
            props.whenLinkButtonTouched = () => this.whenFeatureListWidgetLinkButtonTouched(field);
            props.whenEditButtonTouched = f => this.whenFeatureListWidgetEditButtonTouched(field, f);
            props.whenUnlinkButtonTouched = f => this.whenFeatureListWidgetUnlinkButtonTouched(field, f);
            props.whenDeleteButtonTouched = f => this.whenFeatureListWidgetDeleteButtonTouched(field, f);
        }

        return controller.view(props)
    }

    //

    async loadFeaturesForList(model: gws.types.IModel, forceReload: boolean = false) {
        let es = this.editState;

        let strExtent = e => (e || []).join();

        let request: gws.api.plugin.edit.action.QueryRequest = {
            modelUids: [model.uid],
            relationDepth: 0,
            keyword: es.searchText[model.uid] || '',
        }

        if (model.loadingStrategy === gws.api.core.FeatureLoadingStrategy.bbox) {
            request.extent = this.map.bbox;
        }

        let fc = es.featureCache[model.uid];

        if (
            !forceReload
            && fc
            && fc.keyword === request.keyword
            && fc.extent === strExtent(request.extent)
        ) {
            return;
        }

        let res = await this.app.server.editQueryFeatures(request);
        let features = model.featureListFromProps(res.features);

        let newMap = {};

        for (let f of features) {
            newMap[f.uid] = f;
        }

        for (let f of this.loadedFeatures(model.uid)) {
            if (f.isDirty) {
                newMap[f.uid] = f;
            }
        }

        this.updateEditState({
            featureCache: {
                ...es.featureCache,
                [model.uid]: {
                    keyword: request.keyword,
                    extent: strExtent(request.extent),
                    features: Object.values(newMap),
                }
            }
        });
    }

    relatedModelForField(field) {
        if (!gws.lib.isEmpty(field.relations))
            return this.app.models.model(field.relations[0].modelUid)

    }

    async loadRelatedFeaturesForField(field) {
        let es = this.editState;
        let model = this.relatedModelForField(field)

        let request: gws.api.plugin.edit.action.QueryRequest = {
            modelUids: [model.uid],
            views: ['title'],
            relationDepth: 0,
            keyword: es.searchText[field.uid] || '',
        }

        let fc = es.featureCache[field.uid];

        if (fc && fc.keyword === request.keyword) {
            return
        }

        let features = [];

        if (model.loadingStrategy === gws.api.core.FeatureLoadingStrategy.all || request.keyword) {
            let res = await this.app.server.editQueryFeatures(request);
            features = model.featureListFromProps(res.features);
        }

        this.updateEditState({
            featureCache: {
                ...es.featureCache,
                [field.uid]: {
                    keyword: request.keyword,
                    extent: '',
                    features,

                }
            }
        });

    }

    loadedFeatures(uid) {
        let es = this.editState;
        let fc = es.featureCache[uid];
        return fc ? fc.features : [];
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
        let model = feature.model;

        let atts = feature.currentAttributes();

        // NB changing geometry saves everything,
        // otherwise we cannot validate the form completely

        /*
        if (onlyGeometry) {
            atts = {
                [feature.geometryName]: atts[feature.geometryName],
                [feature.keyName]: atts[feature.keyName],
            }
        } else {
        */
        for (let [k, v] of gws.lib.entries(atts)) {
            atts[k] = await this.serializeValue(v);
        }

        let featureToWrite = feature.clone();
        featureToWrite.attributes = atts;

        let res = await this.app.server.editWriteFeature({
            modelUid: model.uid,
            layerUid: model.layer.uid,
            feature: featureToWrite.getProps(1),
        }, {binary: true});

        if (res.error) {
            this.showDialog({
                type: 'Error',
                errorText: this.__('editSaveErrorText'),
                errorDetails: res.error.info,
            })
            return false;
        }

        let newState: Partial<EditState> = {
            formErrors: {},
        };

        for (let e of res.validationErrors) {
            let msg = e['message'];
            if (msg.startsWith(DEFAULT_MESSAGE_PREFIX))
                msg = this.__(msg)
            newState.formErrors[e.fieldName] = msg;
        }

        if (!gws.lib.isEmpty(newState.formErrors)) {
            this.updateEditState(newState);
            return false
        }

        return true;
    }

    async createFeature(geometry?: ol.geom.Geometry): Promise<gws.types.IFeature> {
        let model = this.editState.selectedModel;
        let attributes = {}

        if (geometry) {
            attributes[model.geometryName] = this.map.geom2shape(geometry)
        }

        let featureToInit = model.featureWithAttributes(attributes);
        featureToInit.isNew = true;

        let res = await this.app.server.editInitFeature({
            modelUid: model.uid,
            layerUid: model.layer.uid,
            feature: featureToInit.getProps(1)
        });

        let newFeature = model.featureFromProps(res.feature);
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
        // let fm = this.editState.featureMap[feature.modelUid] || {};
        // this.updateEditState({
        //     featureMap: {
        //         ...this.editState.featureMap,
        //         [feature.modelUid]: {
        //             ...fm,
        //             [feature.uid]: feature,
        //         }
        //     }
        // })
    }


    //


    whenWidgetChanged(field: gws.types.IModelField, value: any) {
        let es = this.editState,
            feature = es.selectedFeature;

        if (!feature)
            return;
        console.log('XXXX', field, value)
        es.selectedFeature.editAttribute(field.name, value);
        this.updateEditState()
    }

    whenWidgetEntered(field: gws.types.IModelField, value: any) {
        this.whenWidgetChanged(field, value);
        this.whenFormSaveButtonTouched();
    }

    whenGeometryWidgetNewButtonTouched(field: gws.types.IModelField) {
        this.updateEditState({
            drawFeature: this.editState.selectedFeature,
        });
        this.app.startTool('Tool.Edit.Draw')
    }

    whenGeometryWidgetEditButtonTouched(field: gws.types.IModelField) {
        let feature = this.editState.selectedFeature;
        this.update({
            marker: {
                features: [feature],
                mode: 'zoom',
            }
        })
        this.app.startTool('Tool.Edit.Pointer');
    }

    whenFeatureListWidgetNewButtonTouched(field: gws.types.IModelField) {


        // let relationSelected = async (relation: gws.types.IModelRelation) => {
        //     let es = this.editState;
        //     let attributes = {};
        //     let field = relation.model.getField(relation.fieldName);
        //
        //     if (field.dataType === 'feature') {
        //         attributes[field.name] = es.feature;
        //     }
        //
        //     if (field.dataType === 'featureList') {
        //         attributes[field.name] = [es.feature];
        //     }
        //
        //     let feature = await this.createNewFeature(relation.model.getLayer(), attributes, null);
        //
        //     this.setState({
        //         feature,
        //         prevState: es,
        //         relationFieldName: field.name,
        //     });
        //
        //     this.update({editDialogData: null})
        //
        // }
    }


    async whenFeatureListWidgetLinkButtonTouched(field: gws.types.IModelField) {
        let es = this.editState;

        let whenFeatureTouched = (feature) => {
            let atts = es.selectedFeature.currentAttributes();
            let flist = atts[field.name] || [];
            es.selectedFeature.editAttribute(field.name, [...flist, feature]);
            this.closeDialog();
            this.updateEditState();
        }

        let whenSearchTextChanged = async (val) => {
            this.updateSearchText(field.uid, val);
            await this.loadRelatedFeaturesForField(field);
        }

        await this.loadRelatedFeaturesForField(field);

        this.showDialog({
            type: 'SelectFeature',
            field,
            features: this.loadedFeatures(field.uid),
            whenFeatureTouched,
            whenSearchTextChanged,
        });

    }

    whenFeatureListWidgetEditButtonTouched(field: gws.types.IModelField, feature: gws.types.IFeature) {
    }

    whenFeatureListWidgetUnlinkButtonTouched(field: gws.types.IModelField, feature: gws.types.IFeature) {
    }

    whenFeatureListWidgetDeleteButtonTouched(field: gws.types.IModelField, feature: gws.types.IFeature) {
    }


    //


    updateSearchText(uid: string, val: string) {
        let es = this.editState;

        this.updateEditState({
            searchText: {
                ...es.searchText,
                [uid]: val,
            }
        });

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
