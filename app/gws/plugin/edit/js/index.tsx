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
const GEOMETRY_TIMEOUT = 500;


let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;

const MASTER = 'Shared.Edit';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

interface FeatureCacheElement {
    cacheKey: string;
    features: Array<gws.types.IFeature>,
}

interface EditState {
    selectedModel?: gws.types.IModel;
    selectedFeature?: gws.types.IFeature;
    formErrors?: object;
    drawModel?: gws.types.IModel;
    drawFeature?: gws.types.IFeature;
    prevFeatures: Array<gws.types.IFeature>;
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
    whenSearchChanged: (val: string) => void;
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
        let features = cc.cachedFeatures(dd.field.uid);

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
                whenSearchChanged={dd.whenSearchChanged}
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

class Helper {
    master: Controller;

    constructor(master) {
        this.master = master;
    }

    get editState() {
        return this.master.editState;
    }

    get app() {
        return this.master.app;
    }
}

//

class ModelsTab extends gws.View<ViewProps> {
    render() {
        let cc = _master(this).modelsTabController;
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

class ModelsTabController extends Helper {
    get models() {
        return this.master.models;
    }

    async whenModelListItemTouched(model: gws.types.IModel) {
        this.master.selectModel(model)
    }
}

//

class ListTab extends gws.View<ViewProps> {
    async componentDidMount() {
        console.log('FeaturesTab componentDidMount')
        let cc = _master(this).listTabController;
        await cc.whenMounted()
    }

    // componentDidUpdate() {
    //     console.log('FeaturesTab componentDidUpdate')
    //     let cc = _master(this).listTabController;
    //     cc.whenMounted()
    //
    // }
    //
    render() {
        let cc = _master(this).listTabController;
        let es = this.props.editState;
        let model = es.selectedModel;
        let features = cc.master.cachedFeatures(model.uid);
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
                    controller={cc.master}
                    whenFeatureTouched={f => cc.whenFeatureTouched(f)}
                    whenSearchChanged={val => cc.whenSearchChanged(val)}
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
                        whenTouched={() => cc.whenModelsButtonTouched()}
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
                        whenTouched={() => cc.whenNewButtonTouched()}
                    />}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>

    }
}


class ListTabController extends Helper {
    _searchTimer = 0;

    get models() {
        return this.master.models;
    }

    async whenMounted() {
        await this.loadFeaturesForSelectedModel();
    }


    async whenFeatureTouched(feature: gws.types.IFeature) {
        let loaded = await this.master.loadFeatureForForm(feature);
        if (loaded) {
            this.master.selectFeature(loaded);
            this.master.panToFeature(loaded);
        }
    }

    async whenSearchChanged(val: string) {
        let model = this.master.editState.selectedModel;
        this.master.updateSearchText(model.uid, val);
        clearTimeout(this._searchTimer);
        this._searchTimer = Number(setTimeout(
            () => this.loadFeaturesForSelectedModel(),
            SEARCH_TIMEOUT
        ));
    }


    whenModelsButtonTouched() {
        this.master.unselectModel();
        this.master.updateEditState({prevFeatures: []});
    }

    async whenNewButtonTouched() {
        let feature = await this.master.createFeature(this.master.editState.selectedModel);
        this.master.selectFeature(feature);
    }

    whenDrawButtonTouched() {
        this.master.updateEditState({
            drawModel: this.master.editState.selectedModel,
            drawFeature: null,
        });
        this.master.app.startTool('Tool.Edit.Draw')
    }

    async loadFeaturesForSelectedModel() {
        let es = this.master.editState;
        await this.master.loadFeaturesForList(
            es.selectedModel.uid,
            es.selectedModel,
            es.searchText[es.selectedModel.uid],
        );
    }

}

//

class FormTab extends gws.View<ViewProps> {
    async componentDidMount() {
        console.log('FormTab componentDidMount')
        let cc = _master(this).formTabController;
        await cc.whenMounted()
    }

    async componentDidUpdate() {
        // await _master(this).whenMounted();
    }

    render() {
        let cc = _master(this).formTabController;
        let es = this.props.editState;
        let feature = es.selectedFeature;

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
                                    whenTouched={() => cc.whenSaveButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editResetButton', isDirty && 'isActive')}
                                    tooltip={this.__('editReset')}
                                    whenTouched={() => cc.whenResetButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.lib.cls('editCancelButton')}
                                    tooltip={this.__('editCancel')}
                                    whenTouched={() => cc.whenCancelButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    className="editDeleteButton"
                                    tooltip={this.__('editDelete')}
                                    whenTouched={() => cc.whenDeleteButtonTouched()}
                                />
                            </Cell>
                        </Row>
                    </VRow>
                </VBox>
            </sidebar.TabBody>

        </sidebar.Tab>
    }

}

interface WidgetHelper {
    init(field: gws.types.IModelField, props: gws.types.Dict);
}

class GeometryWidgetHelper extends Helper implements WidgetHelper {
    init(field, props) {
        props.whenNewButtonTouched = () => this.whenNewButtonTouched(field);
        props.whenEditButtonTouched = () => this.whenEditButtonTouched(field);
    }

    whenNewButtonTouched(field: gws.types.IModelField) {
        let sf = this.editState.selectedFeature;
        this.master.updateEditState({
            drawModel: sf.model,
            drawFeature: sf,
        });
        this.app.startTool('Tool.Edit.Draw');
    }

    whenEditButtonTouched(field: gws.types.IModelField) {
        this.master.zoomToFeature(this.editState.selectedFeature);
        this.app.startTool('Tool.Edit.Pointer');
    }
}

class FeatureSelectWidgetHelper extends Helper implements WidgetHelper {
    init(field, props) {
        props.features = this.master.cachedFeatures(field.uid);
    }
}

class FeatureSuggestWidgetHelper extends Helper implements WidgetHelper {
    init(field, props) {
        props.features = this.master.cachedFeatures(field.uid);
        props.searchText = this.editState.searchText[field.uid] || '';
        props.whenSearchChanged = val => this.master.formTabController.whenSearchChanged(field, val);
    }
}

class FeatureListWidgetHelper extends Helper implements WidgetHelper {
    init(field, props) {
        props.whenNewButtonTouched = () => this.whenNewButtonTouched(field);
        props.whenLinkButtonTouched = () => this.whenLinkButtonTouched(field);
        props.whenEditButtonTouched = f => this.whenEditButtonTouched(field, f);
        // props.whenUnlinkButtonTouched = f => this.whenUnlinkButtonTouched(field, f);
        props.whenDeleteButtonTouched = f => this.whenDeleteButtonTouched(field, f);
    }

    async whenRelationSelected(field: gws.types.IModelField, relation: gws.types.IModelRelation) {
        let es = this.editState;
        let sf = es.selectedFeature;
        let attributes = {};
        let relModel = this.master.relatedModelForField(field);
        let relField = relModel.getField(relation.fieldName);

        if (relField.attributeType === gws.api.core.AttributeType.feature) {
            attributes[relField.name] = sf;
        }

        if (relField.attributeType === gws.api.core.AttributeType.featurelist) {
            attributes[relField.name] = [sf];
        }

        this.master.pushFeature(sf);
        this.master.selectFeature(await this.master.createFeature(relModel, attributes, null));
        this.master.closeDialog();
    }

    async whenNewButtonTouched(field: gws.types.IModelField) {
        if (field.relations.length === 1)
            return this.whenRelationSelected(field, field.relations[0]);

        this.master.showDialog({
            type: 'SelectRelation',
            relations: field.relations,
            whenRelationSelected: rel => this.whenRelationSelected(field, rel),
        });
    }

    whenLinkedFeatureSelected(field, feature) {
        let es = this.editState;
        let sf = es.selectedFeature;
        let atts = sf.currentAttributes();
        let flist = this.master.removeFeature(atts[field.name], feature);
        sf.editAttribute(field.name, [feature, ...flist]);
        this.master.closeDialog();
        this.master.updateEditState();
    }

    async whenLinkButtonTouched(field: gws.types.IModelField) {
        await this.master.formTabController.loadRelatedFeatures(field);
        this.master.showDialog({
            type: 'SelectFeature',
            field,
            features: this.master.cachedFeatures(field.uid),
            whenFeatureTouched: feature => this.whenLinkedFeatureSelected(field, feature),
            whenSearchChanged: val => this.master.formTabController.whenSearchChanged(field, val),
        });
    }

    async whenEditButtonTouched(field: gws.types.IModelField, feature: gws.types.IFeature) {
        let es = this.editState;
        let sf = es.selectedFeature;
        let loaded = await this.master.loadFeatureForForm(feature);
        if (loaded) {
            this.master.pushFeature(sf);
            this.master.selectFeature(loaded);
            this.master.panToFeature(loaded);
        }
    }

    whenUnlinkButtonTouched(field: gws.types.IModelField, feature: gws.types.IFeature) {
    }

    async whenDeleteConfirmed(field, feature) {
        await this.master.deleteFeature(feature);

        let es = this.editState;
        let sf = es.selectedFeature;
        let atts = sf.currentAttributes();
        let flist = this.master.removeFeature(atts[field.name], feature);
        sf.editAttribute(field.name, flist);
        this.master.closeDialog();
        this.master.updateEditState();
    }

    whenDeleteButtonTouched(field: gws.types.IModelField, feature: gws.types.IFeature) {
        this.master.showDialog({
            type: 'DeleteFeature',
            feature: feature,
            whenConfirmed: () => this.whenDeleteConfirmed(field, feature),
        })
    }
}


class FormTabController extends Helper {
    _searchTimer = 0;
    widgetHelpers: { [type: string]: WidgetHelper };

    constructor(master) {
        super(master);
        this.widgetHelpers = {
            'geometry': new GeometryWidgetHelper(this.master),
            'featureList': new FeatureListWidgetHelper(this.master),
            'featureSelect': new FeatureSelectWidgetHelper(this.master),
            'featureSuggest': new FeatureSuggestWidgetHelper(this.master),
        }
    }


    async whenMounted() {
        let sf = this.editState.selectedFeature;
        if (!sf)
            return;

        this.master.updateEditState({formErrors: []});

        for (let field of sf.model.fields) {
            let p = field.widgetProps;
            console.log('XXX', field)
            if (p && p.type == 'featureSelect') {
                await this.loadRelatedFeatures(field)
            }
            if (p && p.type == 'featureSuggest') {
                await this.loadRelatedFeatures(field)
            }
        }
    }

    async whenSaveButtonTouched() {
        let es = this.editState;
        let feature = es.selectedFeature;

        let ok = await this.master.saveFeature(feature);
        if (ok) {
            await this.master.closeForm();
        }
    }

    whenDeleteButtonTouched() {
        let sf = this.editState.selectedFeature;

        let whenConfirmed = async () => {
            this.master.closeDialog();
            await this.master.deleteFeature(sf);
            await this.master.closeForm();
        }

        this.master.showDialog({
            type: 'DeleteFeature',
            feature: sf,
            whenConfirmed,
        })
    }

    whenResetButtonTouched() {
        let sf = this.editState.selectedFeature;
        sf.resetEdits();
        this.master.updateEditState();
    }

    async whenCancelButtonTouched() {
        await this.master.closeForm();
    }

    whenSearchChanged(field: gws.types.IModelField, val: string) {
        this.master.updateSearchText(field.uid, val);
        clearTimeout(this._searchTimer);
        this._searchTimer = Number(setTimeout(
            () => this.loadRelatedFeatures(field),
            SEARCH_TIMEOUT));
    }

    whenWidgetChanged(field: gws.types.IModelField, value: any) {
        let es = this.editState;
        if (!es.selectedFeature)
            return;
        es.selectedFeature.editAttribute(field.name, value);
        this.master.updateEditState()
    }

    whenWidgetEntered(field: gws.types.IModelField, value: any) {
        this.whenWidgetChanged(field, value);
        this.whenSaveButtonTouched();
    }

    createWidget(field: gws.types.IModelField, feature: gws.types.IFeature, values: gws.types.Dict): React.ReactElement | null {
        let controller = this.master.widgetControllerForField(field);
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

        if (this.widgetHelpers[p.type]) {
            this.widgetHelpers[p.type].init(field, props);
        }

        return controller.view(props)
    }

    async loadRelatedFeatures(field: gws.types.IModelField) {
        let model = this.master.relatedModelForField(field);
        if (model) {
            return this.master.loadFeaturesForList(
                field.uid,
                model,
                this.editState.searchText[field.uid],
            );
        }
    }

}


//

class SidebarView extends gws.View<ViewProps> {
    render() {
        let es = this.props.editState;

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


class Controller extends gws.Controller {
    uid = MASTER;
    models: Array<gws.types.IModel>;
    editLayer: EditLayer;
    _geometryTimer = 0;

    modelsTabController = new ModelsTabController(this);
    listTabController = new ListTabController(this);
    formTabController = new FormTabController(this);


    async init() {
        await super.init();

        this.models = this.app.models.editableModels();

        this.updateEditState({
            searchText: {},
            prevFeatures: [],
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
            this.updateEditState({
                selectedModel: model,
            })
        }
    }


    async whenPointerDownAtCoordinate(coord: ol.Coordinate) {
        let pt = new ol.geom.Point(coord);
        let models;

        // @TODO should be an option whether to query the selected model only or all of them

        if (this.editState.selectedModel)
            models = [this.editState.selectedModel]
        else
            models = this.models.filter(m => m.layer && m.layer.visible);

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
        let selected = this.editState.selectedFeature;

        if (selected && selected.model === loaded.model && selected.uid === loaded.uid) {
            return;
        }

        this.selectFeature(loaded);
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Edit'});
    }


    async whenModifyEnded(feature: gws.types.IFeature) {
        let save = async () => {
            if (feature.isNew) {
                return;
            }
            let ok = await this.saveFeature(feature, true);
            if (ok) {
                this.map.forceUpdate();
            }
        }
        clearTimeout(this._geometryTimer);
        this._geometryTimer = Number(setTimeout(save, GEOMETRY_TIMEOUT));
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

    closeDialog() {
        this.update({editDialogData: null});
    }

    selectModel(model: gws.types.IModel) {
        this.updateEditState({selectedModel: model});
        this.dropCache(model.uid);
    }

    unselectModel() {
        this.updateEditState({selectedModel: null});
        this.dropCache();
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
            prevFeatures: [],
        })
    }

    pushFeature(feature: gws.types.IFeature) {
        let pf = this.removeFeature(this.editState.prevFeatures, feature);
        this.updateEditState({
            prevFeatures: [...pf, feature],
        });
    }

    async popFeature() {
        let pf = this.editState.prevFeatures;
        if (gws.lib.isEmpty(pf)) {
            return false;
        }

        let loaded = await this.loadFeatureForForm(pf[pf.length - 1]);
        if (loaded) {
            this.updateEditState({
                prevFeatures: pf.slice(0, -1)
            });
            this.selectFeature(loaded);
            return true;
        }

        this.updateEditState({
            prevFeatures: [],
        });
        return false;
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

    async closeForm() {
        let ok = await this.popFeature();
        if (ok) {
            return;
        }
        let sf = this.editState.selectedFeature;
        this.unselectFeature();
        if (sf) {
            this.selectModel(sf.model);
        }
    }

    //

    async loadFeaturesForList(cacheUid: string, model: gws.types.IModel, searchText?: string) {
        let ls = model.loadingStrategy;

        let request: gws.api.plugin.edit.action.QueryRequest = {
            modelUids: [model.uid],
            views: ['title'],
            relationDepth: 0,
            keyword: searchText || '',
        }

        if (ls === gws.api.core.FeatureLoadingStrategy.lazy && !searchText) {
            return [];
        }
        if (ls === gws.api.core.FeatureLoadingStrategy.bbox) {
            request.extent = this.map.bbox;
        }

        let res = await this.app.server.editQueryFeatures(request);
        let loaded = model.featureListFromProps(res.features);
        let fmap = new Map();

        for (let f of loaded) {
            fmap.set(f.uid, f);
        }

        for (let f of this.cachedFeatures(cacheUid)) {
            if (f.isDirty) {
                fmap.set(f.uid, f);
            }
        }

        let features = [...fmap.values()];
        this.addCache(cacheUid, request, features);
        return features;
    }

    async loadFeatureForForm(feature: gws.types.IFeature) {
        let res = await this.app.server.editQueryFeatures({
            modelUids: [feature.model.uid],
            featureUids: [feature.uid],
            resolution: this.map.viewState.resolution,
            relationDepth: 1,
        });

        if (gws.lib.isEmpty(res.features)) {
            return;
        }

        return this.app.models.featureFromProps(res.features[0]);
    }


    addCache(uid, request: gws.api.plugin.edit.action.QueryRequest, features) {
        let cacheKey = JSON.stringify([request.keyword, request.extent]);
        this.updateEditState({
            featureCache: {
                ...this.editState.featureCache,
                [uid]: {cacheKey, features}
            }
        });
    }

    dropCache(uid?: string) {
        if (!uid) {
            this.updateEditState({
                featureCache: {}
            });
        } else {
            let fc = {...this.editState.featureCache};
            delete fc[uid];
            this.updateEditState({
                featureCache: fc
            });
        }
    }

    cachedFeatures(uid) {
        let es = this.editState;
        let fc = es.featureCache[uid];
        return fc ? fc.features : [];
    }

    //

    relatedModelForField(field) {
        if (!gws.lib.isEmpty(field.relations))
            return this.app.models.model(field.relations[0].modelUid)
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
        }, {binary: true});

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

        feature.commitEdits();
        this.dropCache();
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
            feature: feature.getProps(0)
        });

        if (res.error) {
            this.showDialog({
                type: 'Error',
                errorText: this.__('editDeleteErrorText'),
                errorDetails: '',
            })
            return false;
        }

        this.dropCache();
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

    updateSearchText(uid: string, val: string) {
        let es = this.editState;

        this.updateEditState({
            searchText: {
                ...es.searchText,
                [uid]: val,
            }
        });

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
