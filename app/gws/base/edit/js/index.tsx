import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';
import * as toolbar from 'gws/elements/toolbar';
import * as draw from 'gws/elements/draw';
import * as components from 'gws/components';

// see app/gws/base/model/validator.py
const DEFAULT_MESSAGE_PREFIX = 'validationError_';

const SEARCH_TIMEOUT = 500;
const GEOMETRY_TIMEOUT = 2000;

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;

const MASTER = 'Shared.Edit';

function _master(obj: any): Controller {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

interface EditState {
    selectedModel?: gws.types.IModel;
    selectedFeature?: gws.types.IFeature;
    formErrors?: object;
    drawModel?: gws.types.IModel;
    drawFeature?: gws.types.IFeature;
    featureHistory: Array<gws.types.IFeature>;
    relationFieldName?: string;
    searchText: { [modelUid: string]: string };
    featureCache: { [key: string]: Array<gws.types.IFeature> },
    isWaiting: boolean,
}

interface SelectModelDialogData {
    type: 'SelectModel';
    models: Array<gws.types.IModel>;
    whenSelected: (model: gws.types.IModel) => void;
}

interface SelectFeatureDialogData {
    type: 'SelectFeature';
    model: gws.types.IModel;
    field: gws.types.IModelField;
    whenFeatureTouched: (f: gws.types.IFeature) => void;
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
    SelectModelDialogData
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

        if (!feature || !feature.geometry) {
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
        let model = _master(this).editState.drawModel;
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
    withSearch: boolean;
    whenSearchChanged: (val: string) => void;
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
                                whenChanged={val => this.props.whenSearchChanged(val)}
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

class Dialog extends gws.View<ViewProps> {

    render() {
        let dd = this.props.editDialogData;

        if (!dd || !dd.type) {
            return null;
        }

        switch (dd.type) {
            case 'SelectModel':
                return <SelectModelDialog {...this.props} />
            case 'SelectFeature':
                return <SelectFeatureDialog {...this.props} />
            case 'DeleteFeature':
                return <DeleteFeatureDialog {...this.props} />
            case 'Error':
                return <ErrorDialog {...this.props} />
        }
    }
}

class SelectModelDialog extends gws.View<ViewProps> {
    render() {
        let dd = this.props.editDialogData as SelectModelDialogData;
        let cc = _master(this);

        let cancelButton = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

        let items = dd.models.map(model => ({
            value: model.uid,
            text: model.title,
        }));

        return <gws.ui.Dialog
            className="editSelectModelDialog"
            title={this.__('editSelectModelTitle')}
            whenClosed={() => cc.closeDialog()}
            buttons={[cancelButton]}
        >
            <Form>
                <Row>
                    <Cell flex>
                        <gws.ui.List
                            items={items}
                            value={null}
                            whenChanged={v => dd.whenSelected(cc.app.modelRegistry.getModel(v))}
                        />
                    </Cell>
                </Row>
            </Form>
        </gws.ui.Dialog>;
    }
}

class SelectFeatureDialog extends gws.View<ViewProps> {
    async componentDidMount() {
        let cc = _master(this);
        let dd = this.props.editDialogData as SelectFeatureDialogData;
        console.log(dd)
        await cc.featureCache.updateForModel(dd.model);
    }

    render() {
        let cc = _master(this);
        let dd = this.props.editDialogData as SelectFeatureDialogData;
        let searchText = cc.getSearchText(dd.model.uid)
        let features = cc.featureCache.getForModel(dd.model);

        let cancelButton = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

        return <gws.ui.Dialog
            className="editSelectFeatureDialog"
            title={this.__('editSelectFeatureTitle')}
            whenClosed={() => cc.closeDialog()}
            buttons={[cancelButton]}
        >
            <FeatureList
                controller={cc}
                whenFeatureTouched={dd.whenFeatureTouched}
                whenSearchChanged={val => cc.whenFeatureListSearchChanged(dd.model, val)}
                features={features}
                searchText={searchText}
                withSearch={dd.model.supportsKeywordSearch}
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

class ModelsTab extends gws.View<ViewProps> {
    async whenModelListItemTouched(model: gws.types.IModel) {
        _master(this).selectModel(model)
    }

    render() {
        let cc = _master(this);
        let items = cc.models;

        items.sort((a, b) => a.title.localeCompare(b.title));

        if (gws.lib.isEmpty(items)) {
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
                                whenTouched={() => this.whenModelListItemTouched(model)}
                                content={model.title}
                            />}
                            uid={model => model.uid}
                            leftButton={model => <components.list.Button
                                className="editorModelListButton"
                                whenTouched={() => this.whenModelListItemTouched(model)}
                            />}
                        />
                    </VRow>
                </VBox>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class ListTab extends gws.View<ViewProps> {
    async whenFeatureTouched(feature: gws.types.IFeature) {
        let cc = _master(this);
        let loaded = await cc.featureCache.loadOne(feature);
        if (loaded) {
            cc.selectFeature(loaded);
            cc.panToFeature(loaded);
        }
    }

    whenSearchChanged(val) {
        let cc = _master(this);
        let es = cc.editState;
        let sm = es.selectedModel;

        cc.whenFeatureListSearchChanged(sm, val);
    }

    whenModelsButtonTouched() {
        let cc = _master(this);
        cc.unselectModel();
        cc.updateEditState({featureHistory: []});
    }

    async whenNewButtonTouched() {
        let cc = _master(this);
        let feature = await cc.createFeature(cc.editState.selectedModel);
        cc.selectFeature(feature);
    }

    whenDrawButtonTouched() {
        let cc = _master(this);
        cc.updateEditState({
            drawModel: cc.editState.selectedModel,
            drawFeature: null,
        });
        cc.app.startTool('Tool.Edit.Draw')
    }

    async componentDidMount() {
        let cc = _master(this);
        await cc.featureCache.updateForSelectedModel();
    }

    render() {
        let cc = _master(this);
        let es = this.props.editState;
        let model = es.selectedModel;
        let features = cc.featureCache.getForModel(model);
        let hasGeom = !gws.lib.isEmpty(model.geometryName);
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
                    whenFeatureTouched={f => this.whenFeatureTouched(f)}
                    whenSearchChanged={val => this.whenSearchChanged(val)}
                    features={features}
                    searchText={searchText}
                    withSearch={model.supportsKeywordSearch}
                />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        {...gws.lib.cls('editModelsAuxButton')}
                        tooltip={this.__('editModelsAuxButton')}
                        whenTouched={() => this.whenModelsButtonTouched()}
                    />
                    <Cell flex/>
                    {model.canCreate && hasGeom && <sidebar.AuxButton
                        {...gws.lib.cls('editDrawAuxButton', this.props.appActiveTool === 'Tool.Edit.Draw' && 'isActive')}
                        tooltip={this.__('editDrawAuxButton')}
                        whenTouched={() => this.whenDrawButtonTouched()}
                    />}
                    {model.canCreate && <sidebar.AuxButton
                        {...gws.lib.cls('editAddAuxButton')}
                        tooltip={this.__('editAddAuxButton')}
                        whenTouched={() => this.whenNewButtonTouched()}
                    />}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>

    }
}


class FormTab extends gws.View<ViewProps> {
    async whenSaveButtonTouched() {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        let ok = await cc.saveFeature(sf);
        if (ok) {
            await cc.closeForm();
        }
    }

    whenDeleteButtonTouched() {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        cc.showDialog({
            type: 'DeleteFeature',
            feature: sf,
            whenConfirmed: () => this.whenDeleteConfirmed(),
        })
    }

    async whenDeleteConfirmed() {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        let ok = await cc.deleteFeature(sf);
        if (ok) {
            await cc.closeDialog();
            await cc.closeForm();
        }
    }

    whenResetButtonTouched() {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        sf.resetEdits();
        cc.updateEditState();
    }

    async whenCancelButtonTouched() {
        let cc = _master(this);

        await cc.closeForm();
    }

    whenWidgetChanged(field: gws.types.IModelField, value: any) {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        sf.editAttribute(field.name, value);
        cc.updateEditState()
    }

    whenWidgetEntered(field: gws.types.IModelField, value: any) {
        this.whenWidgetChanged(field, value);
        this.whenSaveButtonTouched();
    }

    async initWidget(field: gws.types.IModelField, feature: gws.types.IFeature) {
        let cc = _master(this);

        let controller = cc.widgetControllerForField(field);
        if (!controller)
            return;

        let p = field.widgetProps;
        if (cc.widgetHelpers[p.type]) {
            await cc.widgetHelpers[p.type].init(field);
        }
    }


    createWidget(field: gws.types.IModelField, feature: gws.types.IFeature, values: gws.types.Dict): React.ReactElement | null {
        let cc = _master(this);

        let controller = cc.widgetControllerForField(field);
        if (!controller)
            return null;

        let p = field.widgetProps;

        let props: gws.types.Dict = {
            controller,
            feature,
            field,
            widgetProps: field.widgetProps,
            values,
            whenChanged: val => this.whenWidgetChanged(field, val),
            whenEntered: val => this.whenWidgetEntered(field, val),
        }

        if (cc.widgetHelpers[p.type]) {
            cc.widgetHelpers[p.type].setProps(field, props);
        }

        return controller.view(props)
    }

    async componentDidMount() {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        cc.updateEditState({formErrors: []});

        for (let field of sf.model.fields)
            await this.initWidget(field, sf);
    }

    render() {
        let cc = _master(this);
        let es = cc.editState;
        let feature = es.selectedFeature;

        let values = feature.currentAttributes();

        let widgets = [];
        for (let f of feature.model.fields)
            widgets.push(this.createWidget(f, feature, values));

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
                                    model={feature.model}
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
                                    whenTouched={() => this.whenSaveButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editResetButton', isDirty && 'isActive')}
                                    tooltip={this.__('editReset')}
                                    whenTouched={() => this.whenResetButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editCancelButton')}
                                    tooltip={this.__('editCancel')}
                                    whenTouched={() => this.whenCancelButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    className="editDeleteButton"
                                    tooltip={this.__('editDelete')}
                                    whenTouched={() => this.whenDeleteButtonTouched()}
                                />
                            </Cell>
                        </Row>
                    </VRow>
                </VBox>
            </sidebar.TabBody>

        </sidebar.Tab>
    }

}

class WidgetHelper {
    app: gws.types.IApplication

    constructor(controller: gws.types.IController) {
        this.app = controller.app;
    }

    async init(field: gws.types.IModelField) {
    }

    setProps(field: gws.types.IModelField, props: gws.types.Dict) {
    }
}

class GeometryWidgetHelper extends WidgetHelper {
    setProps(field, props) {
        props.whenNewButtonTouched = () => this.whenNewButtonTouched(field);
        props.whenEditButtonTouched = () => this.whenEditButtonTouched(field);
    }

    whenNewButtonTouched(field: gws.types.IModelField) {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        cc.updateEditState({
            drawModel: sf.model,
            drawFeature: sf,
        });
        cc.app.startTool('Tool.Edit.Draw');
    }

    whenEditButtonTouched(field: gws.types.IModelField) {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        cc.zoomToFeature(sf);
        cc.app.startTool('Tool.Edit.Pointer');
    }
}

class FeatureSelectWidgetHelper extends WidgetHelper {
    async init(field) {
        let cc = _master(this);
        await cc.featureCache.updateRelatableForField(field)
    }

    setProps(field, props) {
        let cc = _master(this);
        props.features = cc.featureCache.getRelatableForField(field);

    }
}

class FeatureSuggestWidgetHelper extends WidgetHelper {
    async init(field) {
        let cc = _master(this);
        await cc.featureCache.updateRelatableForField(field)
    }

    setProps(field, props) {
        let cc = _master(this);

        props.features = cc.featureCache.getRelatableForField(field);
        props.searchText = cc.getSearchText(field.uid);
        props.whenSearchChanged = val => this.whenSearchChanged(field, val);
    }

    whenSearchChanged(field, val: string) {
        let cc = _master(this);
        cc.whenFieldSearchChanged(field, val)
    }
}

class FeatureListWidgetHelper extends WidgetHelper {
    async init(field: gws.types.IModelField) {
    }

    setProps(field: gws.types.IModelField, props) {
        props.whenNewButtonTouched = () => this.whenNewButtonTouched(field);
        props.whenLinkButtonTouched = () => this.whenLinkButtonTouched(field);
        props.whenEditButtonTouched = r => this.whenEditButtonTouched(field, r);
        props.whenUnlinkButtonTouched = f => this.whenUnlinkButtonTouched(field, f);
        // props.whenDeleteButtonTouched = r => this.whenDeleteButtonTouched(field, r);
    }

    async whenNewButtonTouched(field: gws.types.IModelField) {
        let cc = _master(this);
        let relatedModels = field.relatedModels();

        if (relatedModels.length === 1) {
            return this.whenModelForNewSelected(field, relatedModels[0]);
        }
        cc.showDialog({
            type: 'SelectModel',
            models: relatedModels,
            whenSelected: model => this.whenModelForNewSelected(field, model),
        });
    }

    async whenLinkButtonTouched(field: gws.types.IModelField) {
        let cc = _master(this);
        let relatedModels = field.relatedModels();

        if (relatedModels.length === 1) {
            return this.whenModelForLinkSelected(field, relatedModels[0]);
        }
        cc.showDialog({
            type: 'SelectModel',
            models: relatedModels,
            whenSelected: model => this.whenModelForLinkSelected(field, model),
        });
    }


    async whenModelForNewSelected(field: gws.types.IModelField, model: gws.types.IModel) {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        let res = await this.app.server.editInitRelatedFeature({
            modelUid: sf.model.uid,
            feature: sf.getProps(),
            fieldName: field.name,
            relatedModelUid: model.uid,
        });

        let newFeature = cc.app.modelRegistry.featureFromProps(res.feature);
        newFeature.isNew = true;
        newFeature.whenSaved = (f) => field.addRelatedFeature(sf, f);

        cc.pushFeature(sf);
        cc.selectFeature(newFeature);
        await cc.closeDialog();
    }


    async whenModelForLinkSelected(field: gws.types.IModelField, model: gws.types.IModel) {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        await cc.featureCache.updateForModel(model);

        cc.showDialog({
            type: 'SelectFeature',
            model: model,
            field,
            whenFeatureTouched: r => this.whenLinkedFeatureSelected(field, r),
        });

    }

    whenLinkedFeatureSelected(field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        field.addRelatedFeature(sf, relatedFeature);

        cc.closeDialog();
        cc.updateEditState();
    }

    whenUnlinkButtonTouched(field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;

        field.removeRelatedFeature(sf, relatedFeature);

        cc.closeDialog();
        cc.updateEditState();
    }


    async whenEditButtonTouched(field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = _master(this);
        let es = cc.editState;
        let sf = es.selectedFeature;
        let loaded = await cc.featureCache.loadOne(relatedFeature);
        if (loaded) {
            cc.pushFeature(sf);
            cc.selectFeature(loaded);
            cc.panToFeature(loaded);
        }
    }

    whenDeleteButtonTouched(field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = _master(this);
        cc.showDialog({
            type: 'DeleteFeature',
            feature: relatedFeature,
            whenConfirmed: () => this.whenDeleteConfirmed(field, relatedFeature),
        })
    }

    async whenDeleteConfirmed(field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = _master(this);

        let ok = await cc.deleteFeature(relatedFeature);

        if (ok) {
            let es = cc.editState;
            let sf = es.selectedFeature;
            let atts = sf.currentAttributes();
            let flist = cc.removeFeature(atts[field.name], relatedFeature);
            sf.editAttribute(field.name, flist);
        }

        await cc.closeDialog();
    }

}

class SidebarView extends gws.View<ViewProps> {
    render() {
        let es = this.props.editState;

        if (es.isWaiting)
            return <gws.ui.Loader/>;

        if (es.selectedFeature)
            return <FormTab {...this.props} />;

        if (es.selectedModel)
            return <ListTab {...this.props} />;

        return <ModelsTab {...this.props} />;
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

class FeatureCache {
    app: gws.types.IApplication

    constructor(controller: gws.types.IController) {
        this.app = controller.app;
    }

    getForModel(model): Array<gws.types.IFeature> {
        let cc = _master(this);
        let es = cc.editState;
        let key = 'model:' + model.uid
        return this.get(key)
    }

    async updateForSelectedModel() {
        let cc = _master(this);
        let es = cc.editState;

        if (es.selectedModel) {
            await this.updateForModel(es.selectedModel)
        }
    }

    async updateForModel(model) {
        let cc = _master(this);
        let es = cc.editState;
        let features = await this.loadMany(model, cc.getSearchText(model.uid));
        let key = 'model:' + model.uid;
        this.checkAndStore(key, features);
    }

    getRelatableForField(field) {
        let key = 'field:' + field.uid;
        return this.get(key);
    }

    async updateRelatableForField(field: gws.types.IModelField) {
        let cc = _master(this);
        let searchText = cc.getSearchText(field.uid);

        let request: gws.api.base.edit.action.GetRelatableFeaturesRequest = {
            modelUid: field.model.uid,
            fieldName: field.name,
            keyword: searchText || '',
            extent: cc.map.bbox,
        }
        let res = await cc.app.server.editGetRelatableFeatures(request);
        let features = cc.app.modelRegistry.featureListFromProps(res.features);
        let key = 'field:' + field.uid;
        this.checkAndStore(key, features);
    }

    async loadMany(model, searchText) {
        let cc = _master(this);
        let ls = model.loadingStrategy;

        let request: gws.api.base.edit.action.GetFeaturesRequest = {
            modelUids: [model.uid],
            keyword: searchText || '',
        }

        if (ls === gws.api.core.FeatureLoadingStrategy.lazy && !searchText) {
            return [];
        }
        if (ls === gws.api.core.FeatureLoadingStrategy.bbox) {
            request.extent = cc.map.bbox;
        }

        let res = await cc.app.server.editGetFeatures(request);
        return model.featureListFromProps(res.features);
    }

    async loadOne(feature: gws.types.IFeature) {
        let cc = _master(this);

        let res = await cc.app.server.editGetFeature({
            modelUid: feature.model.uid,
            featureUid: feature.uid,
        });

        return cc.app.modelRegistry.featureFromProps(res.feature);
    }

    checkAndStore(key: string, features: Array<gws.types.IFeature>) {
        let cc = _master(this);

        let fmap = new Map();

        for (let f of features) {
            fmap.set(f.uid, f);
        }

        for (let f of this.get(key)) {
            if (f.isDirty) {
                fmap.set(f.uid, f);
            }
        }

        this.store(key, [...fmap.values()]);
    }

    drop(uid: string) {
        let cc = _master(this);

        let es = cc.editState;
        let fc = {...es.featureCache};
        delete fc[uid];
        cc.updateEditState({
            featureCache: fc
        });
    }

    clear() {
        let cc = _master(this);

        cc.updateEditState({
            featureCache: {}
        });
    }

    store(key, features) {
        let cc = _master(this);

        let es = cc.editState;
        cc.updateEditState({
            featureCache: {
                ...(es.featureCache || {}),
                [key]: features,
            }
        });
    }

    get(key: string): Array<gws.types.IFeature> {
        let cc = _master(this);

        let es = cc.editState;
        return es.featureCache[key] || [];
    }
}

class Controller extends gws.Controller {
    uid = MASTER;
    editLayer: EditLayer;
    models: Array<gws.types.IModel>

    widgetHelpers: { [key: string]: WidgetHelper } = {
        'geometry': new GeometryWidgetHelper(this),
        'featureList': new FeatureListWidgetHelper(this),
        'featureSelect': new FeatureSelectWidgetHelper(this),
        'featureSuggest': new FeatureSuggestWidgetHelper(this),
    }

    featureCache = new FeatureCache(this);

    async init() {
        await super.init();

        this.updateEditState({
            searchText: {},
            featureHistory: [],
            featureCache: {},
        });

        this.models = [];
        let res = await this.app.server.editGetModels({});
        for (let props of res.models) {
            this.models.push(this.app.modelRegistry.addModel(props));
        }

        if (gws.lib.isEmpty(this.models)) {
            return;
        }

        this.editLayer = this.map.addServiceLayer(new EditLayer(this.map, {
            uid: '_edit',
        }));
        this.editLayer.controller = this;

        this.app.whenCalled('editModel', args => {
            this.selectModel(args.model);
            this.update({
                sidebarActiveTab: 'Sidebar.Edit',
            });
        });

        this.app.whenChanged('mapViewState', () =>
            this.whenMapStateChanged());
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, StoreKeys));
    }

    //

    async whenMapStateChanged() {
        let es = this.editState;
        let model = es.selectedModel;

        if (!model)
            return;

        if (model.loadingStrategy == gws.api.core.FeatureLoadingStrategy.bbox) {
            await this.featureCache.updateForSelectedModel();
        }
    }

    _searchTimer = 0

    async whenFeatureListSearchChanged(model: gws.types.IModel, val: string) {
        this.updateSearchText(model.uid, val);
        clearTimeout(this._searchTimer);
        this._searchTimer = Number(setTimeout(
            () => this.featureCache.updateForModel(model),
            SEARCH_TIMEOUT
        ));
    }


    async whenFieldSearchChanged(field: gws.types.IModelField, val: string) {
        this.updateSearchText(field.uid, val);
        clearTimeout(this._searchTimer);
        this._searchTimer = Number(setTimeout(
            () => this.featureCache.updateRelatableForField(field),
            SEARCH_TIMEOUT
        ));
    }


    async whenPointerDownAtCoordinate(coord: ol.Coordinate) {
        let pt = new ol.geom.Point(coord);
        let models;

        // @TODO should be an option whether to query the selected model only or all of them

        if (this.editState.selectedModel)
            models = [this.editState.selectedModel]
        else
            models = this.models.filter(m => !m.layer || !m.layer.hidden);

        let res = await this.app.server.editGetFeatures({
            shapes: [this.map.geom2shape(pt)],
            modelUids: models.map(m => m.uid),
            resolution: this.map.viewState.resolution,
        });

        if (gws.lib.isEmpty(res.features)) {
            this.unselectFeature();
            console.log('whenPointerDownAtCoordinate: no feature')
            return;
        }

        let loaded = this.app.modelRegistry.featureFromProps(res.features[0]);
        let sf = this.editState.selectedFeature;

        if (sf && sf.model === loaded.model && sf.uid === loaded.uid) {
            console.log('whenPointerDownAtCoordinate: same feature')
            return;
        }

        this.selectFeature(loaded);
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Edit'});

        console.log('whenPointerDownAtCoordinate: found feature', loaded)

    }

    _geometrySaveTimer = 0;


    async whenModifyEnded(feature: gws.types.IFeature) {
        let save = async () => {
            if (feature.isNew) {
                return;
            }

            console.log('whenModifyEnded: begin save')
            this.app.stopTool('Tool.Edit.Pointer');

            let ok = await this.saveFeature(feature, true);
            if (ok) {
                await this.featureCache.updateForSelectedModel();
                // await this.cache.updateRelatableFeatures(feature);
            }

            this.app.startTool('Tool.Edit.Pointer');
            console.log('whenModifyEnded: end save')
        }

        let schedule = () => {
            clearTimeout(this._geometrySaveTimer);
            this._geometrySaveTimer = Number(setTimeout(save, GEOMETRY_TIMEOUT));
        }

        schedule();
    }

    async whenDrawEnded(oFeature: ol.Feature) {
        let df = this.editState.drawFeature;

        if (df) {
            df.setGeometry(oFeature.getGeometry());
            this.selectFeature(df);
        } else {
            let feature = await this.createFeature(
                this.editState.drawModel,
                null,
                oFeature.getGeometry()
            )
            this.selectFeature(feature);
        }
        this.updateEditState({
            drawModel: null,
            drawFeature: null,
        });
    }

    whenDrawCancelled() {
        this.updateEditState({
            drawModel: null,
            drawFeature: null,
        });
        this.app.stopTool('Tool.Edit.Draw');
    }


    //

    showDialog(dd: DialogData) {
        this.update({editDialogData: dd})
    }

    async closeDialog() {
        this.updateEditState({isWaiting: true});
        await gws.lib.sleep(2);
        this.update({editDialogData: null});
        this.updateEditState({isWaiting: false});
    }

    async closeForm() {
        this.updateEditState({isWaiting: true});
        await gws.lib.sleep(2);

        let ok = await this.popFeature();

        if (!ok) {
            let sf = this.editState.selectedFeature;
            this.unselectFeature();
            if (sf) {
                this.selectModel(sf.model);
            }
        }

        this.updateEditState({isWaiting: false});
    }

    //

    selectModel(model: gws.types.IModel) {
        this.updateEditState({selectedModel: model});
        this.featureCache.drop(model.uid);
    }

    unselectModel() {
        this.updateEditState({selectedModel: null});
        this.featureCache.clear();
    }

    selectFeature(feature: gws.types.IFeature) {
        this.updateEditState({selectedFeature: feature});
        if (feature.geometry) {
            this.app.startTool('Tool.Edit.Pointer');
        }
    }

    unselectFeature() {
        this.updateEditState({
            selectedFeature: null,
            featureHistory: [],
        })
    }

    pushFeature(feature: gws.types.IFeature) {
        let hist = this.removeFeature(this.editState.featureHistory, feature);
        this.updateEditState({
            featureHistory: [...hist, feature],
        });
    }

    async popFeature() {
        let hist = this.editState.featureHistory || [];
        if (hist.length === 0) {
            return false;
        }

        let last = hist[hist.length - 1];
        this.updateEditState({
            featureHistory: hist.slice(0, -1)
        });
        this.selectFeature(last);
        return true;
        //
        //
        // let loaded = await this.featureCache.loadOne(hist[hist.length - 1]);
        // if (loaded) {
        //     this.updateEditState({
        //         featureHistory: hist.slice(0, -1)
        //     });
        //     this.selectFeature(loaded);
        //     return true;
        // }
        //
        // this.updateEditState({
        //     featureHistory: [],
        // });
        // return false;
    }

    panToFeature(feature) {
        if (!feature.geometry)
            return;
        this.update({
            marker: {
                features: [feature],
                mode: 'pan',
            }
        })
    }

    zoomToFeature(feature) {
        if (!feature.geometry)
            return;
        this.update({
            marker: {
                features: [feature],
                mode: 'zoom',
            }
        })
    }

    widgetControllerForField(field: gws.types.IModelField): gws.types.IModelWidget | null {
        let p = field.widgetProps;
        if (!p)
            return null;

        let tag = 'ModelWidget.' + p.type;
        return (
            this.app.controllerByTag(tag) ||
            this.app.createControllerFromConfig(this, {tag})
        ) as gws.types.IModelWidget;
    }


    //


    //

    getModel(modelUid) {
        return this.app.modelRegistry.getModel(modelUid)
    }

    removeFeature(flist: Array<gws.types.IFeature>, feature: gws.types.IFeature) {
        return (flist || []).filter(f => f.model !== feature.model || f.uid !== feature.uid);
    }

    //

    async saveFeature(feature: gws.types.IFeature, onlyGeometry = false) {
        let model = feature.model;

        let atts = feature.currentAttributes();

        // NB changing geometry saves everything,
        // otherwise we cannot validate the form completely

        for (let [k, v] of gws.lib.entries(atts)) {
            atts[k] = await this.serializeValue(v);
        }

        let featureToWrite = feature.clone();
        featureToWrite.attributes = atts;

        let res = await this.app.server.editWriteFeature({
            modelUid: model.uid,
            feature: featureToWrite.getProps(1),
        }, {binary: false});

        if (res.error) {
            this.showDialog({
                type: 'Error',
                errorText: this.__('editSaveErrorText'),
                errorDetails: '',
            })
            return false;
        }

        let formErrors = {};
        for (let e of res.validationErrors) {
            let msg = e['message'];
            if (msg.startsWith(DEFAULT_MESSAGE_PREFIX))
                msg = this.__(msg)
            formErrors[e.fieldName] = msg;
        }

        if (!gws.lib.isEmpty(formErrors)) {
            this.updateEditState({formErrors});
            return false
        }

        let savedFeature = this.app.modelRegistry.featureFromProps(res.feature);
        feature.copyFrom(savedFeature)

        feature.whenSaved(feature);

        this.featureCache.clear();
        this.map.forceUpdate();

        return true;
    }

    async createFeature(model: gws.types.IModel, attributes?: object, geometry?: ol.geom.Geometry): Promise<gws.types.IFeature> {
        attributes = attributes || {};

        if (geometry) {
            attributes[model.geometryName] = this.map.geom2shape(geometry)
        }

        let featureToInit = model.featureWithAttributes(attributes);
        featureToInit.isNew = true;

        let res = await this.app.server.editInitFeature({
            modelUid: model.uid,
            feature: featureToInit.getProps(1),
        });

        let newFeature = model.featureFromProps(res.feature);
        newFeature.isNew = true;

        return newFeature;
    }

    async deleteFeature(feature: gws.types.IFeature) {

        let res = await this.app.server.editDeleteFeature({
            modelUid: feature.model.uid,
            feature: feature.getProps()
        });

        if (res.error) {
            this.showDialog({
                type: 'Error',
                errorText: this.__('editDeleteErrorText'),
                errorDetails: '',
            })
            return false;
        }

        this.featureCache.clear();
        this.map.forceUpdate();

        return true;
    }

    async serializeValue(val) {
        if (!val)
            return val;

        if (val instanceof FileList) {
            let content: Uint8Array = await gws.lib.readFile(val[0]);
            return {name: val[0].name, content}
        }

        return val;
    }

    //

    updateSearchText(key: string, val: string) {
        let es = this.editState;

        this.updateEditState({
            searchText: {
                ...es.searchText,
                [key]: val,
            }
        });
    }

    getSearchText(key: string) {
        let es = this.editState;
        return es.searchText[key] || '';
    }


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
