import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from './sidebar';
import * as toolbar from './toolbar';
import * as draw from './draw';

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;

const MASTER = 'Shared.Edit';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}

interface EditDialogData {
    name: string;
    relations?: Array<gws.types.IModelRelation>;
    features?: Array<gws.types.IFeature>;
    layers?: Array<gws.types.IFeatureLayer>;
    featureToDelete?: gws.types.IFeature;
    error?: string;
    errorDetails?: string;

    whenRelationSelected?: (r: gws.types.IModelRelation) => void;
    whenFeatureSelected?: (f: gws.types.IFeature) => void;
    whenConfirmed?: () => void;


}

interface EditState {
    layer?: gws.types.IFeatureLayer;
    feature?: gws.types.IFeature;
    formValues?: object;
    formErrors?: object;
    drawFeature?: gws.types.IFeature;
    prevState?: EditState;
    relationFieldName?: string;
}


interface EditViewProps extends gws.types.ViewProps {
    editState: EditState;
    editDialogData?: EditDialogData;
    editUpdateCount: number;
    editSearchText?: string;
    editErrors: gws.types.Dict;
    mapUpdateCount: number;
    appActiveTool: string;
}

const EditStoreKeys = [
    'editState',
    'editDialogData',
    'editUpdateCount',
    'editSearchText',
    'editErrors',
    'mapUpdateCount',
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

//

export class EditPointerTool extends gws.Tool {
    oFeatureCollection: ol.Collection<ol.Feature>;
    snap: boolean = true;

    selectFeature(feature: gws.types.IFeature) {
        if (this.oFeatureCollection) {
            this.oFeatureCollection.clear();
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
            if (feature && cc.isFeatureEditable(feature)) {
                found = feature;
            }
            // if (feature && feature.layer === cc.activeLayer) {
            //     found = feature;
            // }
        });

        if (found) {
            this.selectFeature(found);
            cc.whenPointerDownAtFeature(found);
            return true;
        }

        cc.whenPointerDownAtCoordinate(evt.coordinate);
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
                        cc.whenModifyEnded(feature)


                    }
                }
            }
        });

        let ixs: Array<ol.interaction.Interaction> = [ixPointer, ixModify];

        if (this.snap && cc.activeLayer)
            ixs.push(this.map.snapInteraction({
                    layer: cc.activeLayer
                })
            );

        this.map.appendInteractions(ixs);

    }

    stop() {
        this.oFeatureCollection = null;
    }

}


class EditDrawTool extends draw.Tool {
    whenEnded(shapeType, oFeature) {
        _master(this).whenDrawToolEnded(oFeature);
    }

    enabledShapes() {
        let la = _master(this).activeLayer;
        if (!la)
            return null;
        return ENABLED_SHAPES_BY_TYPE[la.geometryType.toUpperCase()];
    }

    whenCancelled() {
        _master(this).whenDrawToolCancelled();
    }
}


class FeatureDetailsTab extends gws.View<EditViewProps> {
    render() {
        let cc = _master(this);
        let es = cc.editState;
        let isDirty = es.feature.isNew || es.feature.isDirty;
        let formValues = {
            ...es.feature.attributes,
            ...es.feature.editedAttributes,
        };

        return <sidebar.Tab className="modEditSidebar modEditSidebarFormTab">
            <sidebar.TabHeader>
                <Row>
                    <Cell flex>
                        <gws.ui.Title content={cc.featureTitle(es.feature)}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <VBox>
                    <VRow flex>
                        <Cell flex>
                            <Form tabular>
                                <gws.components.Form
                                    controller={this.props.controller}
                                    feature={es.feature}
                                    values={formValues}
                                    model={cc.map.models.getModelForLayer(es.feature.layer)}
                                    errors={es.formErrors}
                                    whenChanged={cc.whenFormChanged.bind(cc)}
                                    whenEntered={cc.whenFormEntered.bind(cc)}
                                    whenEvent={cc.whenFormEvent.bind(cc)}
                                />
                            </Form>
                        </Cell>
                    </VRow>
                    <VRow>
                        <Row>
                            <Cell flex/>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.tools.cls('modEditSaveButton', isDirty && 'isActive')}
                                    tooltip={this.__('modEditSave')}
                                    whenTouched={() => cc.whenFormSaveButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.tools.cls('modEditResetButton', isDirty && 'isActive')}
                                    tooltip={this.__('modEditReset')}
                                    whenTouched={() => cc.whenFormResetButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    {...gws.tools.cls('modEditCancelButton')}
                                    tooltip={this.__('modEditCancel')}
                                    whenTouched={() => cc.whenFormCancelButtonTouched()}
                                />
                            </Cell>
                            <Cell spaced>
                                <gws.ui.Button
                                    className="modEditDeleteButton"
                                    tooltip={this.__('modEditDelete')}
                                    whenTouched={() => cc.whenFormDeleteButtonTouched()}
                                />
                            </Cell>
                        </Row>
                    </VRow>
                </VBox>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        {...gws.tools.cls('modEditGotoLayersAuxButton')}
                        tooltip={this.__('modEditGotoLayersAuxButton')}
                        whenTouched={() => cc.whenGotoLayersButtonTouched()}
                    />
                    <sidebar.AuxButton
                        {...gws.tools.cls('modEditGotoFeaturesAuxButton')}
                        tooltip={this.__('modEditGotoFeaturesAuxButton')}
                        whenTouched={() => cc.whenGotoFeaturesButtonTouched()}
                    />
                    <Cell flex/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }

}

class FeatureListTab extends gws.View<EditViewProps> {
    render() {
        let cc = _master(this);
        let layer = cc.activeLayer;
        let features = [...layer.features.sort(
            (a, b) => cc.featureTitle(a).localeCompare(cc.featureTitle(b)))
        ];

        let search = (this.props.editSearchText || '').trim().toLowerCase();

        if (search) {
            features = features.filter(f => f.elements.title.toLowerCase().includes(search))
        }

        let hasGeom = Boolean(cc.map.models.getModelForLayer(layer).geometryName)

        let searchBox = <div className="modSearchBox">
            <Row>
                <Cell>
                    <gws.ui.Button className='modSearchIcon'/>
                </Cell>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={this.__('modSearchPlaceholder')}
                        withClear={true}
                        value={this.props.editSearchText}
                        whenChanged={v => cc.update({editSearchText: v})}
                    />
                </Cell>
            </Row>
        </div>;

        let isEmpty = !layer.shouldDraw || features.length === 0;

        return <sidebar.Tab className="modEditSidebar">

            <sidebar.TabHeader>
                <Row>
                    <Cell>
                        <gws.ui.Title content={layer.title}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>


            {isEmpty && <sidebar.EmptyTabBody>
                <gws.ui.Text content={this.__('modEditNoFeature')}/>
            </sidebar.EmptyTabBody>}

            {!isEmpty && <sidebar.TabBody>
                <gws.components.feature.List
                    controller={cc}
                    features={features}
                    content={f => <gws.ui.Link
                        content={cc.featureTitle(f)}
                        whenTouched={() => cc.whenFeatureListNameTouched(f)}
                    />}
                    leftButton={f => <gws.components.list.Button
                        className="modEditorLayerListButton"
                        whenTouched={() => cc.whenFeatureListNameTouched(f)}
                    />}
                />
            </sidebar.TabBody>}

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        {...gws.tools.cls('modEditGotoLayersAuxButton')}
                        tooltip={this.__('modEditGotoLayersAuxButton')}
                        whenTouched={() => cc.whenGotoLayersButtonTouched()}
                    />
                    <Cell flex/>
                    <sidebar.AuxButton
                        {...gws.tools.cls('modEditAddAuxButton')}
                        tooltip={this.__('modEditAddAuxButton')}
                        whenTouched={() => cc.whenFeatureListNewButtonTouched()}
                    />
                    {hasGeom && <sidebar.AuxButton
                        {...gws.tools.cls('modEditDrawAuxButton', this.props.appActiveTool === 'Tool.Edit.Draw' && 'isActive')}
                        tooltip={this.__('modEditDrawAuxButton')}
                        whenTouched={() => cc.whenFeatureListDrawButtonTouched()}
                    />}
                    {/*{hasGeom && <sidebar.AuxButton*/}
                    {/*    {...gws.tools.cls('modEditPointerAuxButton', this.props.appActiveTool === 'Tool.Edit.Pointer' && 'isActive')}*/}
                    {/*    tooltip={this.__('modEditPointerAuxButton')}*/}
                    {/*    whenTouched={() => cc.whenSelectButtonTouched()}*/}
                    {/*/>}*/}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}


class LayerListTab extends gws.View<EditViewProps> {
    render() {
        let cc = this.props.controller.app.controller(MASTER) as Controller;
        let layers = cc.editableLayers;


        layers = [...layers]
            .filter(a => a.visible)
            .sort((a, b) => a.title.localeCompare(b.title));

        if (gws.tools.empty(layers)) {
            return <sidebar.EmptyTab>
                {this.__('modEditNoLayer')}
            </sidebar.EmptyTab>;
        }

        return <sidebar.Tab className="modEditSidebar">
            <sidebar.TabHeader>
                <Row>
                    <Cell>
                        <gws.ui.Title content={this.__('modEditTitle')}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.List
                    controller={this.props.controller}
                    items={layers}
                    content={la => <gws.ui.Link
                        whenTouched={() => cc.whenLayerNameTouched(la)}
                        content={la.title}
                    />}
                    uid={la => la.uid}
                    leftButton={la => <gws.components.list.Button
                        className="modEditorLayerListButton"
                        whenTouched={() => cc.whenLayerNameTouched(la)}
                    />}
                />
            </sidebar.TabBody>
        </sidebar.Tab>
    }

}

class EditSidebarView extends gws.View<EditViewProps> {
    render() {
        let cc = _master(this);
        let es = cc.editState;

        if (es.feature)
            return <FeatureDetailsTab {...this.props} />;

        if (es.layer)
            return <FeatureListTab {...this.props} />;

        return <LayerListTab {...this.props} />;
    }
}

class EditSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modEditSidebarIcon';

    get tooltip() {
        return this.__('modEditSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(EditSidebarView, EditStoreKeys)
        );
    }
}

class EditDialog extends gws.View<EditViewProps> {

    render() {
        let dd = this.props.editDialogData;

        if (!dd || !dd.name) {
            return null;
        }

        let cc = _master(this);

        let cancelButton = <gws.ui.Button className="cmpButtonFormCancel" whenTouched={() => cc.closeDialog()}/>;

        if (dd.name === 'SelectRelation') {
            let relations = dd.relations;
            let items = relations.map((r, n) => ({
                value: n,
                text: r.title,
            }));

            return <gws.ui.Dialog
                className="modEditSelectRelationDialog"
                title={'Objekt-Typ auswählen'}
                whenClosed={() => cc.closeDialog()}
                buttons={[cancelButton]}
            >
                <Form>
                    <Row>
                        <Cell flex>
                            <gws.ui.List
                                items={items}
                                value={null}
                                whenChanged={v => dd.whenRelationSelected(relations[v])}
                            />
                        </Cell>
                    </Row>
                </Form>
            </gws.ui.Dialog>;
        }


        if (dd.name === 'SelectFeature') {

            let layers = dd.layers;

            let features = [];
            for (let la of layers) {
                features = features.concat(la.features);
            }

            return <gws.ui.Dialog
                className="modEditSelectFeatureDialog"
                title={'Objekt auswählen'}
                whenClosed={() => cc.closeDialog()}
                buttons={[cancelButton]}
            >
                <Form>
                    <Row>
                        <Cell flex>
                            <gws.components.feature.List
                                controller={cc}
                                features={features}
                                content={f => <gws.ui.Link
                                    content={cc.featureTitle(f)}
                                    whenTouched={() => dd.whenFeatureSelected(f)}
                                />}
                                withZoom
                            />
                        </Cell>
                    </Row>
                </Form>
            </gws.ui.Dialog>;
        }

        if (dd.name === 'ConfirmDeleteFeature') {

            let feature = dd.featureToDelete;

            return <gws.ui.Confirm
                title={'Objekt löschen'}
                text={'Object "' + cc.featureTitle(feature) + '" löschen?"'}
                whenConfirmed={() => dd.whenConfirmed()}
                whenRejected={() => cc.closeDialog()}
            />
        }

        if (dd.name === 'ErrorWhenSave') {

            return <gws.ui.Alert
                title={'Fehler'}
                error={dd.error}
                details={dd.errorDetails}
                whenClosed={() => cc.closeDialog()}
            />
        }

        return null;

    }
}


class EditToolbarButton extends toolbar.Button {
    iconClass = 'modEditToolbarButton';
    tool = 'Tool.Edit.Pointer';

    get tooltip() {
        return this.__('modEditToolbarButton');
    }

}


class Controller extends gws.Controller {
    uid = MASTER;
    selectedStyle: gws.types.IStyle;
    editableLayers: Array<gws.types.IFeatureLayer>;

    async init() {
        await super.init();

        let res1 = await this.app.server.editGetModels({});
        this.map.loadModels(res1.models || []);

        this.editableLayers = [];
        let res2 = await this.app.server.editGetLayers({});
        if (res2.layers) {
            for (let props of res2.layers) {
                let la = this.map.getLayer(props.uid);
                if (la)
                    this.editableLayers.push(la as gws.types.IFeatureLayer);
            }
        }


        this.app.whenCalled('editLayer', args => {
            this.update({
                sidebarActiveTab: 'Sidebar.Edit',
            })
            this.setState({layer: args.layer})
        });


    }

    get appOverlayView() {
        return this.createElement(
            this.connect(EditDialog, EditStoreKeys));
    }

    closeDialog() {
        this.update({editDialogData: null});
    }


    //

    whenLayerNameTouched(layer) {
        this.setState({layer});
    }

    whenSelectButtonTouched() {
    }

    whenGotoLayersButtonTouched() {
        this.setState({});
    }

    whenGotoFeaturesButtonTouched() {
        this.setState({layer: this.activeLayer});
    }


    //

    whenFeatureListNameTouched(feature: gws.types.IFeature) {
        this.reloadAndActivateFeature(feature, 'pan');
    }

    whenFeatureListDrawButtonTouched() {
        let es = this.editState;
        this.setState({
            ...es,
            drawFeature: null,
        });
        this.app.startTool('Tool.Edit.Draw')
    }

    async whenFeatureListNewButtonTouched() {
        let feature = await this.createNewFeature(this.activeLayer, {}, null);
        this.activateFeature(feature, 'pan');
    }

    //

    whenFormSaveButtonTouched() {
        let es = this.editState;
        if (es.feature)
            this.saveFeature(es.feature);

    }

    whenFormResetButtonTouched() {
        let es = this.editState;
        es.feature.editedAttributes = {};
        this.setState({...es});
    }

    whenFormCancelButtonTouched() {
        this.popState();
    }

    whenFormDeleteButtonTouched() {
        this.deleteFeature(this.activeFeature, true);
    }


    //


    async whenFormChanged(field: gws.types.IModelField, value) {
        let es = this.editState;
        let v = es.feature.getAttribute(field.name);

        if (es.feature.getAttribute(field.name) === value) {
            delete es.feature.editedAttributes[field.name];
        } else {
            es.feature.editedAttributes[field.name] = value;
        }

        this.setState({
            ...es,
        });

    }

    async whenFormEntered(field: gws.types.IModelField) {
        this.whenFormSaveButtonTouched()
    }


    async whenFormEvent(field: gws.types.IModelField, type, opts) {

        let es = this.editState;

        switch (type) {

            case 'relation.new': {
                let relationSelected = async (relation: gws.types.IModelRelation) => {
                    let es = this.editState;
                    let attributes = {};
                    let field = relation.model.getField(relation.fieldName);

                    if (field.dataType === 'feature') {
                        attributes[field.name] = es.feature;
                    }

                    if (field.dataType === 'featureList') {
                        attributes[field.name] = [es.feature];
                    }

                    let feature = await this.createNewFeature(relation.model.getLayer(), attributes, null);

                    this.setState({
                        feature,
                        prevState: es,
                        relationFieldName: field.name,
                    });

                    this.update({editDialogData: null})

                }

                if (field.relations.length === 1)
                    return relationSelected(field.relations[0]);

                this.update({
                    editDialogData: {
                        name: 'SelectRelation',
                        relations: field.relations,
                        whenRelationSelected: relationSelected,
                    }
                })

                return;
            }


            case 'relation.link': {

                let featureSelected = (feature) => {
                    let flist = es.feature.getEditedAttribute(field.name) || [];
                    this.whenFormChanged(field, flist.concat(feature));
                    this.closeDialog();
                }

                this.update({
                    editDialogData: {
                        name: 'SelectFeature',
                        layers: field.relations.map(r => r.model.getLayer()),
                        whenFeatureSelected: featureSelected,
                    }
                });

                return;
            }

            case 'relation.unlink': {
                let sel: gws.types.IFeature = opts.selectedFeature;
                let flist = es.feature.getEditedAttribute(field.name) || [];
                this.whenFormChanged(field, flist.fliter(feature => !sel.isSame(feature)));
                return;
            }

            case 'relation.edit': {
                await this.reloadAndActivateFeature(opts.selectedFeature, 'pan');
                this.setState({
                    ...this.editState,
                    prevState: es,
                });
                return;
            }

            case 'relation.delete': {
                this.setState({
                    ...this.editState,
                    prevState: es,
                    relationFieldName: field.name,
                });
                await this.deleteFeature(opts.selectedFeature, true);
                return;
            }


            case 'geometry.draw':
                this.setState({
                    ...es,
                    drawFeature: es.feature,
                });
                this.app.startTool('Tool.Edit.Draw')
                return;


            case 'geometry.edit':
                this.focusFeature(es.feature, 'pan');
                this.app.startTool('Tool.Edit.Pointer');
                return;

            case 'file.view':
                let url = [
                    '/_/cmd/editHttpGetPath',
                    '/projectUid/' + this.app.project.uid,
                    '/layerUid/' + es.feature.layer.uid,
                    '/featureUid/' + es.feature.uid,
                    '/fieldName/' + field.name
                ].join('');

                let fileName = es.feature.getAttribute(field.name);

                gws.tools.downloadUrl(url, fileName, null);


        }


    }


    //

    async whenDrawToolEnded(oFeature: ol.Feature) {
        let es = this.editState;
        let feature = es.drawFeature;
        let geom = oFeature.getGeometry();

        this.app.startTool('Tool.Edit.Pointer');

        if (feature) {
            feature.setGeometry(geom);
        } else {
            feature = await this.createNewFeature(this.activeLayer, {}, geom);
        }

        this.activateFeature(feature, '');

    }

    whenDrawToolCancelled() {
        if (this.activeLayer)
            this.app.startTool('Tool.Edit.Pointer')
    }

    whenPointerDownAtFeature(feature) {
        if (feature !== this.activeFeature)
            this.reloadAndActivateFeature(feature, '');
    }

    async whenPointerDownAtCoordinate(coord: ol.Coordinate) {
        let pt = new ol.geom.Point(coord);
        let res = await this.app.server.editQueryFeatures({
            shapes: [this.map.geom2shape(pt)],
            // layerUids: this.activeLayer ? [this.activeLayer.uid] : [],
            layerUids: [],
            resolution: this.map.viewState.resolution,
        });

        let flist = this.map.featureListFromProps(res.features);
        if (flist.length > 0) {
            let feature = flist[0];
            feature.layer.addFeature(feature);
            await this.activateFeature(feature, 'pan');
            return true;
        }

        let layer = this.activeLayer;

        if (layer && layer.loadingStrategy === 'single')
            layer.clear();

        this.setState({layer});
        this.app.map.focusFeature(null);
        return false;
    }

    geomTimer: any = 0;

    whenModifyEnded(feature) {
        clearTimeout(this.geomTimer);
        this.geomTimer = setTimeout(() => this.saveFeature(feature, true), 500);
    }


    //


    //

    get activeLayer(): gws.types.IFeatureLayer | null {
        if (this.editState.layer)
            return this.editState.layer;
        if (this.editState.feature)
            return this.editState.feature.layer;
    }

    get activeFeature(): gws.types.IFeature | null {
        return this.editState.feature;
    }

    //

    async createNewFeature(layer: gws.types.IFeatureLayer, attributes, geometry) {
        let model = this.map.models.getModelForLayer(layer);

        if (geometry) {
            attributes = {
                ...attributes,
                [model.geometryName]: this.map.geom2shape(geometry)
            }
        }

        let feNew = this.map.featureFromProps({attributes});
        feNew.layer = layer;
        feNew.model = model;

        let res = await this.app.server.editInitFeatures({
            features: [feNew.getProps(1)]
        });

        let feature = this.map.featureFromProps(res.features[0]);

        // if (!feature.oFeature && model.geometryName)
        //     feature.oFeature = new ol.Feature();

        feature.isNew = true;
        layer.addFeature(feature);
        feature.redraw();

        return feature;

    }

    //

    async reloadFeature(feature: gws.types.IFeature) {
        feature.layer.removeFeature(feature);
        let feUpdated = await this.loadFeature(feature);
        feature.updateFrom(feUpdated);
        feature.layer.addFeature(feature);
    }

    async reloadAndActivateFeature(feature: gws.types.IFeature, markerMode = '') {
        await this.reloadFeature(feature);
        this.activateFeature(feature, markerMode);
    }

    activateFeature(feature: gws.types.IFeature, markerMode = '') {
        this.focusFeature(feature, markerMode);
        this.setState({feature});
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Edit'});
    }

    //

    async deleteFeature(feature: gws.types.IFeature, withConfirm) {

        let confirmed = async () => {
            this.closeDialog();

            let layer = feature.layer;

            let res = await this.app.server.editDeleteFeatures({
                features: [feature.getProps(0)]
            });

            if (res.error) {
                this.update({
                    editDialogData: {
                        name: 'ErrorWhenSave',
                        error: "Daten konnten nicht gespeichert werden",
                        errorDetails: res.error.info,
                    }
                })
                return;
            }

            layer.removeFeature(feature);
            this.popState();
            this.map.forceUpdate();
        }

        if (!withConfirm)
            return confirmed();

        this.update({
            editDialogData: {
                name: 'ConfirmDeleteFeature',
                featureToDelete: feature,
                whenConfirmed: confirmed,
            },
        });
    }

    async serializeValue(val) {
        if (!val)
            return val;


        if (val.type === 'FileWidgetValue') {
            if (val.files) {
                let files: FileList = val.files;
                let content: Uint8Array = await gws.tools.readFile(val.files[0]);
                return {
                    type: 'File',
                    name: val.files[0].name,
                    content
                }
            }
        }

        return val;
    }

    async saveFeature(feature: gws.types.IFeature, onlyGeometry = false) {
        let feSave = new gws.map.Feature(this.map);

        feSave.updateFrom(feature);

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
            for (let [k, v] of gws.tools.entries(atts)) {
                atts[k] = await this.serializeValue(v);
            }
        }

        feSave.attributes = atts;

        let res = await this.app.server.editWriteFeatures({
            features: [feSave.getProps(onlyGeometry ? 0 : 1)]
        }, {binary: true});


        if (res.error) {
            this.update({
                editDialogData: {
                    name: 'ErrorWhenSave',
                    error: "Daten konnten nicht gespeichert werden",
                    errorDetails: res.error.info,
                }
            })
            return;
        }

        let es = this.editState;
        let newState = {...es};

        newState.formErrors = null;

        let props = res.features[0];

        if (props.errors) {
            newState.formErrors = {};
            for (let e of props.errors) {
                newState.formErrors[e['name']] = e['error'];
            }
            this.setState(newState);
            return;
        }

        let feUpdated = this.map.featureFromProps(props);

        feature.layer.removeFeature(feature);
        feature.updateFrom(feUpdated);

        if (onlyGeometry) {
            delete feature.editedAttributes[feature.geometryName];
        } else {
            feature.editedAttributes = {};
        }

        feature.layer.addFeature(feature);
        feature.redraw();

        // newState.formValues = feature.attributes;

        if (es.prevState)
            this.popState()
        else
            this.setState(newState);

        this.map.forceUpdate();
    }


    //


    focusFeature(feature: gws.types.IFeature, markerMode) {
        if (markerMode) {
            this.update({
                marker: {
                    features: [feature],
                    mode: markerMode,
                }
            })
        }

        this.app.map.focusFeature(feature);

    }

    async loadFeature(feature: gws.types.IFeature) {
        let model = this.map.models.getModelForLayer(feature.layer);

        let res = await this.app.server.editReadFeatures({
            features: [{
                layerUid: String(feature.layer.uid),
                attributes: {
                    [model.keyName]: String(feature.uid)
                },
                keyName: model.keyName,
            }]
        });

        return this.map.featureFromProps(res.features[0]);
    }


    isFeatureEditable(feature: gws.types.IFeature) {
        let ok = this.editableLayers.indexOf(feature.layer) >= 0;
        return ok;
    }

    //

    featureTitle(f: gws.types.IFeature) {
        return f.elements.title || (this.__('modEditNewObjectName'));
    }

    //

    get editState(): EditState {
        return this.getValue('editState') || {};
    }

    setState(es: EditState) {
        this.update({editState: es});
        this.map.focusFeature(es.feature);
    }

    async popState() {
        let es = this.editState;
        let ps = es.prevState || {layer: this.activeLayer};

        if (ps.feature) {
            let atts = ps.feature.attributes;
            await this.reloadFeature(ps.feature);

            // when going back to the 'parent' feature,
            // we need to preserve scalars, but update possible feature lists

            if (es.relationFieldName) {
                delete ps.feature.editedAttributes[es.relationFieldName];
            }
        }

        this.setState({...ps});
    }

}

export const tags = {
    'Shared.Edit': Controller,
    'Sidebar.Edit': EditSidebar,
    'Tool.Edit.Pointer': EditPointerTool,
    'Tool.Edit.Draw': EditDrawTool,
    'Toolbar.Edit': EditToolbarButton,

};
